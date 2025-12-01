import random
import traceback
import numpy as np
from deap import base, creator, tools, algorithms
from core.database import supabase

# --- 1. Setup DEAP ---
# สร้างคลาสสำหรับ Fitness และ Individual เพียงครั้งเดียว
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# --- 2. Constants & Configuration ---
DAYS = 5
SLOTS_PER_DAY = 10  # 08:00 - 17:00 (รวมพักเที่ยง)
LUNCH_SLOT = 4      # Slot 4 = 12:00 - 13:00

# Config: เพิ่มโหมด 'precise' สำหรับการจัดตารางที่ซับซ้อนและเงื่อนไขเยอะ
GEN_CONFIGS = {
    # แนะนำให้ใช้โหมดนี้เพื่อให้ตรงเงื่อนไขทั้ง 17 ข้อมากที่สุด
    'precise':  {
        'pop_size': 2000,       # ประชากรเยอะ เพื่อความหลากหลาย
        'generations': 500,     # รอบเยอะ เพื่อให้ AI เกลี่ยงานครูได้ละเอียด
        'runs': 1,
        'mutation_prob': 0.2
    },
    'balanced': {'pop_size': 800, 'generations': 200, 'runs': 1, 'mutation_prob': 0.3},
    'fast':     {'pop_size': 200, 'generations': 50,  'runs': 1, 'mutation_prob': 0.4}
}

# --- 3. Helper Functions ---
def get_course_metadata(course):
    """วิเคราะห์ข้อมูลวิชา เพื่อระบุเงื่อนไขพิเศษ"""
    subj = course.get('subjects', {}) or {}
    if isinstance(subj, list): subj = subj[0] 
    
    t_hrs = int(subj.get('theory_hours') or 0)
    p_hrs = int(subj.get('practice_hours') or 0)
    total_hours = t_hrs + p_hrs
    duration = total_hours if total_hours > 0 else 1
    
    subj_name = str(subj.get('subject_name', '')).strip()
    
    # 1. เงื่อนไขข้อ 7: วิชาลูกเสือ
    is_scout = 'ลูกเสือ' in subj_name or 'scout' in subj_name.lower()
    
    # 2. เงื่อนไขข้อ 16: วิชาคอมพิวเตอร์ (บังคับห้อง LB)
    comp_targets = [
        "การเขียนโปรแกรมคอมพิวเตอร์",
        "การพัฒนาโปรแกรมบนอุปกรณ์พกพา",
        "ไมโครคอนโทรลเลอร์",
        "วงจรพัลส์และดิจิทัล",
        "อุปกรณ์อิเล็กทรอนิกส์และวงจร",
        "การใช้โปรแกรมคอมพิวเตอร์กราฟิก"
    ]
    is_computer_subj = any(target in subj_name for target in comp_targets)
    
    # 3. วิชาทฤษฎีบังคับ (บังคับห้อง TH) - เพิ่มเติมเพื่อความเป็นระเบียบ
    theory_targets = [
        "ภาษาไทย", "ภาษาอังกฤษ", "วิทยาศาสตร์", "คณิตศาสตร์คอมพิวเตอร์"
    ]
    is_theory_subj = any(target in subj_name for target in theory_targets)
    
    # เงื่อนไขข้อ 17: ครูที่ปรึกษา (ถ้าใน DB มีข้อมูล advisor_id ให้ return ค่ามาใช้)
    advisor_id = course.get('advisor_id') 
    
    return duration, is_scout, is_computer_subj, is_theory_subj, advisor_id

def find_stadium_index(room_ids):
    """ค้นหาห้องที่เป็นสนาม (สำหรับลูกเสือ)"""
    for idx, r_code in enumerate(room_ids):
        code_lower = r_code.lower()
        if any(x in code_lower for x in ['สนาม', 'stadium', 'field', 'sport', 'foot', 'ball']):
            return idx
    return len(room_ids) - 1 # Fallback ไปห้องสุดท้ายถ้าหาไม่เจอ

# --- 4. Smart Initialization (หัวใจสำคัญ: หาช่องว่างก่อนลง) ---
def create_smart_individual(courses, room_count, allowed_teachers_map, room_ids):
    ind = [None] * len(courses)
    stadium_idx = find_stadium_index(room_ids)
    
    # เตรียม Index ห้องสำหรับวิชาเฉพาะทาง
    comp_rooms = [i for i, r in enumerate(room_ids) if r in ['LB101', 'LB102']]
    theory_rooms = [i for i, r in enumerate(room_ids) if r in ['TH201', 'TH202']]
    # Fallback กันเหนียว
    if not comp_rooms: comp_rooms = [0]
    if not theory_rooms: theory_rooms = [0]

    # ตารางบันทึกการจองชั่วคราว (เพื่อกันชนตั้งแต่เริ่ม)
    used_teacher_slots = set() # (abs_slot, teacher_id)
    used_room_slots = set()    # (abs_slot, room_idx)
    used_student_slots = set() # (abs_slot, group_id)

    # สุ่มลำดับวิชาที่จะลงตาราง
    indices = list(range(len(courses)))
    random.shuffle(indices)

    for i in indices:
        course = courses[i]
        duration, is_scout, is_comp_subj, is_theory_subj, _ = get_course_metadata(course)
        
        # Group ID สำหรับเช็คนักเรียนชนกัน (ข้อ 3)
        dept = course.get('department')
        yr = course.get('year_level')
        grp = course.get('group_no', '1')
        group_id = f"{dept}_{yr}_{grp}"

        # สุ่มครูจากผู้ที่มีสิทธิ์สอน (ข้อ 15)
        valid_teachers = allowed_teachers_map.get(i, [0])
        teacher_idx = random.choice(valid_teachers) if valid_teachers else 0

        # --- Case 1: วิชาลูกเสือ (Fixed Slot) ---
        if is_scout:
            # ข้อ 7: พุธ 15.00-17.00 (Day 2, Slot 7) -> Index 27
            final_slot = 27
            room_idx = stadium_idx
            
            # จำเป็นต้องลง แม้จะชน (เพราะเป็นกฎตายตัว)
            for t in range(duration):
                curr = final_slot + t
                used_teacher_slots.add((curr, teacher_idx))
                used_room_slots.add((curr, room_idx))
                used_student_slots.add((curr, group_id))
            
            ind[i] = [room_idx, final_slot, teacher_idx]
            continue

        # --- Case 2: วิชาทั่วไป/เฉพาะทาง (หาช่องว่าง) ---
        
        # เลือกกลุ่มห้องเป้าหมาย
        candidate_rooms = list(range(room_count))
        if is_comp_subj: candidate_rooms = comp_rooms
        elif is_theory_subj: candidate_rooms = theory_rooms
        
        random.shuffle(candidate_rooms) # สุ่มห้องในกลุ่มเพื่อกระจายตัว
        
        found_placement = False
        
        # วนหาห้องและเวลาที่ว่างพร้อมกัน
        for room_idx in candidate_rooms:
            # สร้างรายการเวลาที่เป็นไปได้ (เว้นพักเที่ยง)
            possible_starts = []
            for d in range(DAYS):
                for s in range(SLOTS_PER_DAY - duration + 1):
                    # ข้อ 10: ห้ามทับพักเที่ยง
                    if s <= LUNCH_SLOT < s + duration: continue
                    # ข้อ 9: ไม่เกิน 17.00
                    if s + duration > 9: continue 
                    # ข้อ 13: เช็ควันว่างครูเมธา (คร่าวๆ) ตรงนี้ก็ได้ หรือไปรอ Mutation ก็ได้
                    # แต่เพื่อความเร็ว ปล่อยให้ Penalty จัดการเรื่อง Soft Constraint
                    
                    possible_starts.append((d * SLOTS_PER_DAY) + s)
            
            random.shuffle(possible_starts)

            for start_slot in possible_starts:
                # ตรวจสอบการชนกัน (Look-ahead Check)
                collision = False
                for t in range(duration):
                    curr = start_slot + t
                    
                    # ข้อ 6: ห้องว่างไหม?
                    if (curr, room_idx) in used_room_slots: 
                        collision = True; break
                    # ข้อ 4,5: ครูว่างไหม?
                    if (curr, teacher_idx) in used_teacher_slots:
                        collision = True; break
                    # ข้อ 3: นักเรียนว่างไหม?
                    if (curr, group_id) in used_student_slots:
                        collision = True; break
                
                if not collision:
                    # เจอที่ว่าง! จองเลย
                    for t in range(duration):
                        curr = start_slot + t
                        used_room_slots.add((curr, room_idx))
                        used_teacher_slots.add((curr, teacher_idx))
                        used_student_slots.add((curr, group_id))
                    
                    ind[i] = [room_idx, start_slot, teacher_idx]
                    found_placement = True
                    break
            
            if found_placement: break
        
        # ถ้าหาที่ลงไม่ได้จริงๆ (หายากมากถ้าห้องพอ) -> จำใจต้องสุ่มลงไปก่อน
        if not found_placement:
            fallback_room = candidate_rooms[0]
            d = random.randint(0, DAYS - 1)
            s = random.randint(0, 8)
            if s >= LUNCH_SLOT: s+=1
            if s+duration > SLOTS_PER_DAY: s = SLOTS_PER_DAY - duration
            final_slot = (d * SLOTS_PER_DAY) + s
            ind[i] = [fallback_room, final_slot, teacher_idx]

    return creator.Individual(ind)

# --- 5. Mutation (ปรับปรุงเพื่อรักษากฎ) ---
def smart_mutate(individual, courses, room_count, allowed_teachers_map, indpb=0.2):
    for i, gene in enumerate(individual):
        _, is_scout, is_comp_subj, is_theory_subj, _ = get_course_metadata(courses[i])
        
        if is_scout: continue # ห้ามแตะต้องลูกเสือเด็ดขาด
        
        # Mutate Room: ห้ามเปลี่ยนประเภทห้องของวิชาบังคับ
        if random.random() < indpb:
            if not is_comp_subj and not is_theory_subj:
                gene[0] = random.randint(0, room_count - 1)
            # ถ้าเป็น Comp/Theory เราไม่เปลี่ยนห้องใน Mutation เพื่อรักษา Hard Constraint
        
        # Mutate Time: ลองขยับเวลา
        if random.random() < indpb: 
            d = random.randint(0, DAYS - 1)
            candidates = [0, 1, 2, 3, 5, 6, 7]
            s = random.choice(candidates)
            duration, _, _, _, _ = get_course_metadata(courses[i])
            if s + duration > SLOTS_PER_DAY: s = SLOTS_PER_DAY - duration
            gene[1] = (d * SLOTS_PER_DAY) + s
            
        # Mutate Teacher: เปลี่ยนครู (ในรายชื่อที่สอนได้)
        if random.random() < indpb: 
            valid = allowed_teachers_map.get(i, [])
            if valid: gene[2] = random.choice(valid)
                
    return individual,

# --- 6. Fitness Function (High Penalty) ---
def evaluate(individual, courses, room_ids, instructor_ids, 
             room_details, instructor_details_map, head_instructor_ids):
    penalty = 0
    
    room_usage = {}
    teacher_usage = {}
    student_usage = {}
    
    teacher_hours = {tid: 0 for tid in instructor_ids}
    teacher_days_active = {tid: set() for tid in instructor_ids}

    for i, gene in enumerate(individual):
        r_idx, start_slot, t_idx = gene
        room_code = room_ids[r_idx]
        teacher_id = instructor_ids[t_idx]
        course = courses[i]
        
        duration, is_scout, is_comp_subj, is_theory_subj, advisor_id = get_course_metadata(course)
        
        dept = course.get('department')
        yr = course.get('year_level')
        grp = course.get('group_no', '1')
        group_id = f"{dept}_{yr}_{grp}"
        
        teacher_obj = instructor_details_map.get(teacher_id, {})
        
        day = start_slot // SLOTS_PER_DAY
        slot = start_slot % SLOTS_PER_DAY
        end_slot = slot + duration

        # --- Hard Constraints Checks (โทษประหาร 1 ล้าน) ---
        
        # ข้อ 16: ห้องคอม
        if is_comp_subj and room_code not in ['LB101', 'LB102']:
            penalty += 1_000_000
        
        # ห้องทฤษฎีบังคับ
        if is_theory_subj and room_code not in ['TH201', 'TH202']:
            penalty += 1_000_000
            
        # ข้อ 7: ลูกเสือ
        if is_scout:
            if day != 2 or slot != 7: penalty += 1_000_000
            if not any(x in room_code.lower() for x in ['สนาม', 'stadium', 'field']):
                 penalty += 500_000 
            # ข้อ 17: ครูที่ปรึกษา (ถ้ามีข้อมูล)
            if advisor_id and int(advisor_id) != int(teacher_id):
                 penalty += 200_000

        # ข้อ 10: พักเที่ยง
        for t in range(duration):
            curr_slot_in_day = slot + t
            if curr_slot_in_day == LUNCH_SLOT:
                 penalty += 1_000_000
        
        # --- Soft Constraints & Collisions ---

        # ข้อ 13: ครูเมธา
        if 'เมธา' in teacher_obj.get('first_name', ''):
            if day == 0 and slot < 4: penalty += 50_000
            if day == 4 and slot >= 5: penalty += 50_000

        # ข้อ 9: เลิกเย็น
        if end_slot > 9: penalty += 100_000

        # Loop เช็คการชนกัน
        for t in range(duration):
            curr_abs = start_slot + t
            
            teacher_hours[teacher_id] += 1
            teacher_days_active[teacher_id].add(day)
            
            # ข้อ 6: ห้องชน
            if (curr_abs, r_idx) in room_usage: penalty += 1_000_000
            else: room_usage[(curr_abs, r_idx)] = True
            
            # ข้อ 4,5: ครูชน
            if (curr_abs, teacher_id) in teacher_usage: penalty += 1_000_000
            else: teacher_usage[(curr_abs, teacher_id)] = True
                
            # ข้อ 3: นร.ชน
            if (curr_abs, group_id) in student_usage: penalty += 1_000_000
            else: student_usage[(curr_abs, group_id)] = True

    # --- Summary Checks (Workload) ---
    hours_values = []
    for tid in instructor_ids:
        h = teacher_hours[tid]
        dept = instructor_details_map.get(tid, {}).get('department', '')
        
        # ข้อ 1: หัวหน้าสอน 18-24
        if tid in head_instructor_ids:
            if h < 18 or h > 24: penalty += 50_000 * abs(h - 21) # ยิ่งห่าง 21 ยิ่งโดนปรับ
        # ข้อ 2: ครูทั่วไป >= 18
        elif h < 18:
            penalty += 20_000 * (18 - h)
            
        # ข้อ 12: ครูคอมสอนทุกวัน
        if 'คอม' in str(dept) or 'computer' in str(dept).lower():
            if len(teacher_days_active[tid]) < 5:
                penalty += 10_000 * (5 - len(teacher_days_active[tid]))
        
        if h > 0: hours_values.append(h)

    # ข้อ 14: เกลี่ยชั่วโมง (SD)
    if hours_values:
        penalty += (np.std(hours_values) * 5000)

    return (penalty,)

# --- 7. Main Execution ---
def run_genetic_algorithm(mode='balanced'):
    print(f"🧬 AI SCHEDULER STARTED... MODE: {mode.upper()}")
    cfg = GEN_CONFIGS.get(mode, GEN_CONFIGS['balanced'])

    try:
        # Load Data from Supabase
        courses = supabase.table('curriculums').select("*, subjects(*)").execute().data
        rooms = supabase.table('classrooms').select("*").execute().data
        instructors = supabase.table('instructors').select("*").execute().data
        
        if not courses or not rooms or not instructors:
            return {"status": "error", "message": "Incomplete Data"}

        # Prepare Maps & IDs
        room_ids = [r['room_code'] for r in rooms]
        instructor_ids = [i['id'] for i in instructors]
        room_details = {r['room_code']: r.get('room_type', '') for r in rooms}
        instructor_details_map = {i['id']: i for i in instructors}
        
        # Map Names to IDs
        instructor_name_map = {
            (ins['first_name'].strip(), ins['last_name'].strip()): int(ins['id']) 
            for ins in instructors
        }
        instructor_db_id_to_index = {int(ins['id']): idx for idx, ins in enumerate(instructors)}
        
        # Identify Heads
        head_instructor_ids = set()
        for ins in instructors:
            pos = str(ins.get('position_role', '')).lower()
            if 'head' in pos or 'หัวหน้า' in pos:
                head_instructor_ids.add(ins['id'])
        
        # Map Allowed Teachers per Course
        allowed_teachers_map = {} 
        for idx, course in enumerate(courses):
            valid_indices = []
            subj_data = course.get('subjects')
            if isinstance(subj_data, list) and subj_data: subj_data = subj_data[0]
            
            if subj_data:
                for k in range(1, 6): 
                    fname = subj_data.get(f'instructor_{k}_fname')
                    lname = subj_data.get(f'instructor_{k}_lname')
                    if fname and lname:
                        key = (fname.strip(), lname.strip())
                        if key in instructor_name_map:
                            real_id = instructor_name_map[key]
                            if real_id in instructor_db_id_to_index:
                                valid_indices.append(instructor_db_id_to_index[real_id])
            
            if not valid_indices: 
                valid_indices = list(range(len(instructors)))
            allowed_teachers_map[idx] = valid_indices

        # Register DEAP functions
        for alias in ['individual', 'population', 'evaluate', 'mutate', 'mate', 'select']:
            if hasattr(toolbox, alias): toolbox.unregister(alias)

        toolbox.register("individual", create_smart_individual, 
                         courses=courses, room_count=len(room_ids),
                         allowed_teachers_map=allowed_teachers_map, room_ids=room_ids)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", smart_mutate, 
                         courses=courses, room_count=len(room_ids), 
                         allowed_teachers_map=allowed_teachers_map, indpb=cfg['mutation_prob']) 
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate, 
                         courses=courses, room_ids=room_ids, instructor_ids=instructor_ids,
                         room_details=room_details, instructor_details_map=instructor_details_map,
                         head_instructor_ids=head_instructor_ids)

        # Run Evolution
        best_overall = None
        best_overall_fitness = float('inf')

        for run_idx in range(cfg['runs']):
            print(f"   🔄 Run {run_idx+1}/{cfg['runs']}")
            pop = toolbox.population(n=cfg['pop_size'])
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=cfg['mutation_prob'],     
                                           ngen=cfg['generations'], stats=stats, halloffame=hof, verbose=True)
            
            current_best = hof[0]
            fit = current_best.fitness.values[0]
            print(f"      ✅ Score: {fit:,.0f}")
            if fit < best_overall_fitness:
                best_overall = current_best
                best_overall_fitness = fit

        print(f"🏆 FINAL BEST FITNESS: {best_overall_fitness:,.0f}")
        save_to_db(best_overall, courses, room_ids, instructor_ids)
        return {"status": "success", "mode": mode, "penalty": best_overall_fitness}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def save_to_db(best_schedule, courses, room_ids, instructor_ids):
    print("💾 Saving to database...")
    try:
        supabase.table('generated_schedules').delete().neq('id', 0).execute()
        data_list = []
        
        for i, gene in enumerate(best_schedule):
            r_idx, start_slot, t_idx = gene
            course = courses[i]
            duration, _, _, _, _ = get_course_metadata(course)
            
            s_name = "Unknown"
            subj = course.get('subjects')
            if isinstance(subj, list) and subj: subj = subj[0]
            if isinstance(subj, dict): s_name = subj.get('subject_name', 'Unknown')
            
            for t in range(duration):
                current_slot = start_slot + t
                day = current_slot // SLOTS_PER_DAY
                slot_in_day = current_slot % SLOTS_PER_DAY
                
                if slot_in_day == LUNCH_SLOT: continue
                if day != (start_slot // SLOTS_PER_DAY): continue

                record = {
                    "subject_code": course.get('subject_code', 'N/A'),
                    "subject_name": s_name,
                    "room_code": room_ids[r_idx],
                    "instructor_id": int(instructor_ids[t_idx]),
                    "day_of_week": int(day),
                    "start_slot": int(slot_in_day),
                    "department": course.get('department', 'General'),
                    "year_level": course.get('year_level', 'N/A')
                }
                data_list.append(record)
        
        batch_size = 1000
        for k in range(0, len(data_list), batch_size):
            supabase.table('generated_schedules').insert(data_list[k:k+batch_size]).execute()
            
        print(f"✅ Saved {len(data_list)} slots successfully!")
        
    except Exception as e:
        print(f"❌ Error saving to DB: {e}")
        traceback.print_exc()