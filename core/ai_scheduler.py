import random
import traceback
import numpy as np
from deap import base, creator, tools, algorithms
from core.database import supabase

# --- 1. Setup DEAP ---
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# --- 2. Constants ---
DAYS = 5
SLOTS_PER_DAY = 10
LUNCH_SLOT = 4

# Config: เพิ่ม Mutation Rate เพื่อให้ AI หาทางออกจากกรอบที่แคบลงได้
GEN_CONFIGS = {
    'balanced': {'pop_size': 500, 'generations': 100, 'runs': 1, 'mutation_prob': 0.3},
}

# --- 3. Helper Functions ---
def get_course_metadata(course):
    subj = course.get('subjects', {}) or {}
    if isinstance(subj, list): subj = subj[0] 
    
    t_hrs = int(subj.get('theory_hours') or 0)
    p_hrs = int(subj.get('practice_hours') or 0)
    total_hours = t_hrs + p_hrs
    duration = total_hours if total_hours > 0 else 1
    
    subj_name = str(subj.get('subject_name', '')).strip()
    
    # 1. ลูกเสือ
    is_scout = 'ลูกเสือ' in subj_name or 'scout' in subj_name.lower()
    
    # 2. วิชาคอมพิวเตอร์ (บังคับ LB101, LB102)
    comp_targets = [
        "การเขียนโปรแกรมคอมพิวเตอร์",
        "การพัฒนาโปรแกรมบนอุปกรณ์พกพา",
        "ไมโครคอนโทรลเลอร์",
        "วงจรพัลส์และดิจิทัล",
        "อุปกรณ์อิเล็กทรอนิกส์และวงจร",
        "การใช้โปรแกรมคอมพิวเตอร์กราฟิก"
    ]
    is_computer_subj = any(target in subj_name for target in comp_targets)
    
    # 3. วิชาทฤษฎี (บังคับ TH201, TH202)
    theory_targets = [
        "ภาษาไทย",
        "ภาษาอังกฤษ",
        "วิทยาศาสตร์",
        "คณิตศาสตร์คอมพิวเตอร์"
    ]
    is_theory_subj = any(target in subj_name for target in theory_targets)
    
    return duration, is_scout, is_computer_subj, is_theory_subj

def find_stadium_index(room_ids):
    for idx, r_code in enumerate(room_ids):
        code_lower = r_code.lower()
        if any(x in code_lower for x in ['สนาม', 'stadium', 'field', 'sport']):
            return idx
    return len(room_ids) - 1

# --- 4. Initialization ---
def create_smart_individual(courses, room_count, allowed_teachers_map, room_ids):
    ind = [None] * len(courses)
    stadium_idx = find_stadium_index(room_ids)
    
    # หา index ของห้องแต่ละประเภทเตรียมไว้
    comp_rooms = [i for i, r in enumerate(room_ids) if r in ['LB101', 'LB102']]
    theory_rooms = [i for i, r in enumerate(room_ids) if r in ['TH201', 'TH202']]
    
    # Fallback: ถ้าหาไม่เจอ ให้ใช้ห้องแรกไปก่อน (กัน Crash)
    if not comp_rooms: comp_rooms = [0]
    if not theory_rooms: theory_rooms = [0]

    indices = list(range(len(courses)))
    random.shuffle(indices)

    for i in indices:
        course = courses[i]
        duration, is_scout, is_comp_subj, is_theory_subj = get_course_metadata(course)
        valid_teachers = allowed_teachers_map.get(i, [0])
        teacher_idx = random.choice(valid_teachers) if valid_teachers else 0
        
        # --- Logic การเลือกห้องแบบเจาะจง ---
        if is_scout:
            ind[i] = [stadium_idx, 27, teacher_idx] # Wed 15:00
            
        elif is_comp_subj:
            # บังคับห้องคอม (LB)
            room_idx = random.choice(comp_rooms)
            ind[i] = generate_random_slot(room_idx, teacher_idx, duration)
            
        elif is_theory_subj:
            # บังคับห้องทฤษฎี (TH)
            room_idx = random.choice(theory_rooms)
            ind[i] = generate_random_slot(room_idx, teacher_idx, duration)
            
        else:
            # วิชาทั่วไป
            room_idx = random.randint(0, room_count - 1)
            ind[i] = generate_random_slot(room_idx, teacher_idx, duration)

    return creator.Individual(ind)

def generate_random_slot(room_idx, teacher_idx, duration):
    # ฟังก์ชันช่วยสุ่มเวลา (จะได้ไม่ต้องเขียนซ้ำ)
    d = random.randint(0, DAYS - 1)
    s = random.randint(0, 8) 
    if s >= LUNCH_SLOT: s += 1 
    if s + duration > SLOTS_PER_DAY: s = SLOTS_PER_DAY - duration
    final_slot = (d * SLOTS_PER_DAY) + s
    return [room_idx, final_slot, teacher_idx]

# --- 5. Mutation ---
def smart_mutate(individual, courses, room_count, allowed_teachers_map, indpb=0.3):
    for i, gene in enumerate(individual):
        _, is_scout, is_comp_subj, is_theory_subj = get_course_metadata(courses[i])
        
        if is_scout: continue # ห้ามแตะลูกเสือ
        
        # Mutate Room: ห้ามเปลี่ยนห้อง ถ้าเป็นวิชาบังคับ (Comp/Theory)
        if random.random() < indpb:
            if not is_comp_subj and not is_theory_subj:
                gene[0] = random.randint(0, room_count - 1)
        
        # Mutate Time
        if random.random() < indpb: 
            d = random.randint(0, DAYS - 1)
            candidates = [0, 1, 2, 3, 5, 6, 7]
            s = random.choice(candidates)
            duration, _, _, _ = get_course_metadata(courses[i])
            if s + duration > SLOTS_PER_DAY: s = SLOTS_PER_DAY - duration
            gene[1] = (d * SLOTS_PER_DAY) + s
            
        # Mutate Teacher
        if random.random() < indpb: 
            valid = allowed_teachers_map.get(i, [])
            if valid: gene[2] = random.choice(valid)
                
    return individual,

# --- 6. Fitness Function ---
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
        
        duration, is_scout, is_comp_subj, is_theory_subj = get_course_metadata(course)
        
        dept = course.get('department')
        yr = course.get('year_level')
        grp = course.get('group_no', '1')
        group_id = f"{dept}_{yr}_{grp}"
        
        teacher_obj = instructor_details_map.get(teacher_id, {})
        
        day = start_slot // SLOTS_PER_DAY
        slot = start_slot % SLOTS_PER_DAY
        end_slot = slot + duration

        # --- Rule Checks ---
        
        # 1. บังคับห้องคอม (LB101, LB102)
        if is_comp_subj:
            if room_code not in ['LB101', 'LB102']:
                penalty += 1_000_000

        # 2. บังคับห้องทฤษฎี (TH201, TH202) -> เพิ่มใหม่
        if is_theory_subj:
            if room_code not in ['TH201', 'TH202']:
                penalty += 1_000_000

        # 3. ลูกเสือ
        if is_scout:
            if day != 2 or slot != 7: penalty += 1_000_000
            if not any(x in room_code.lower() for x in ['สนาม', 'stadium', 'field']):
                 penalty += 500_000 

        # 4. ครูเมธา
        if 'เมธา' in teacher_obj.get('first_name', ''):
            if day == 0 and slot < 4: penalty += 50_000
            if day == 4 and slot >= 5: penalty += 50_000

        # 5. เลิกเรียนเย็นเกิน
        if end_slot > 9: penalty += 100_000

        for t in range(duration):
            curr_abs = start_slot + t
            curr_slot_in_day = slot + t
            
            # ห้ามพักเที่ยง
            if curr_slot_in_day == LUNCH_SLOT: penalty += 1_000_000
            
            teacher_hours[teacher_id] += 1
            teacher_days_active[teacher_id].add(day)
            
            # Zero Tolerance Collisions
            if (curr_abs, r_idx) in room_usage: penalty += 1_000_000
            else: room_usage[(curr_abs, r_idx)] = True
            
            if (curr_abs, teacher_id) in teacher_usage: penalty += 1_000_000
            else: teacher_usage[(curr_abs, teacher_id)] = True
                
            if (curr_abs, group_id) in student_usage: penalty += 1_000_000
            else: student_usage[(curr_abs, group_id)] = True

    # --- Summary Checks ---
    hours_values = []
    for tid in instructor_ids:
        h = teacher_hours[tid]
        dept = instructor_details_map.get(tid, {}).get('department', '')
        
        if tid in head_instructor_ids:
            if h < 18 or h > 24: penalty += 50_000 * abs(h - 21)
        elif h < 18:
            penalty += 20_000 * (18 - h)
            
        if 'คอม' in str(dept):
            if len(teacher_days_active[tid]) < 5:
                penalty += 10_000 * (5 - len(teacher_days_active[tid]))
        
        if h > 0: hours_values.append(h)

    if hours_values:
        penalty += (np.std(hours_values) * 5000)

    return (penalty,)

# --- 7. Main Execution ---
def run_genetic_algorithm(mode='balanced'):
    print(f"🧬 AI SCHEDULER STARTED... MODE: {mode.upper()}")
    cfg = GEN_CONFIGS.get(mode, GEN_CONFIGS['balanced'])

    try:
        courses = supabase.table('curriculums').select("*, subjects(*)").execute().data
        rooms = supabase.table('classrooms').select("*").execute().data
        instructors = supabase.table('instructors').select("*").execute().data
        
        if not courses or not rooms or not instructors:
            return {"status": "error", "message": "Incomplete Data"}

        room_ids = [r['room_code'] for r in rooms]
        instructor_ids = [i['id'] for i in instructors]
        room_details = {r['room_code']: r.get('room_type', '') for r in rooms}
        instructor_details_map = {i['id']: i for i in instructors}
        
        instructor_name_map = {
            (ins['first_name'].strip(), ins['last_name'].strip()): int(ins['id']) 
            for ins in instructors
        }
        instructor_db_id_to_index = {int(ins['id']): idx for idx, ins in enumerate(instructors)}
        
        head_instructor_ids = set()
        for ins in instructors:
            pos = str(ins.get('position_role', '')).lower()
            if 'head' in pos or 'หัวหน้า' in pos:
                head_instructor_ids.add(ins['id'])
        
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
            duration, _, _, _ = get_course_metadata(course)
            
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