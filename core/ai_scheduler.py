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

# --- 2. Constants & Config ---
DAYS = 5
SLOTS_PER_DAY = 11 
TOTAL_SLOTS = DAYS * SLOTS_PER_DAY
LUNCH_SLOT = 4  # 12:30-13:30 (ห้ามเรียน)

# Config Presets
GEN_CONFIGS = {
    'draft':    {'pop_size': 200, 'generations': 50,  'runs': 1, 'mutation_prob': 0.6},
    'balanced': {'pop_size': 600, 'generations': 200, 'runs': 2, 'mutation_prob': 0.4},
    'perfect':  {'pop_size': 1500, 'generations': 500, 'runs': 5, 'mutation_prob': 0.3}
}

# --- 3. Helper Functions ---
def get_course_info(course):
    """ดึงข้อมูลวิชา + เช็คว่าเป็น Lab หรือไม่"""
    try:
        if course.get('subjects'):
            subj = course['subjects']
            if isinstance(subj, list) and subj: subj = subj[0]
            elif isinstance(subj, dict): pass
            else: return 2, False
            
            t_hrs = int(subj.get('theory_hours') or 0)
            p_hrs = int(subj.get('practice_hours') or 0)
            total = t_hrs + p_hrs
            is_lab = p_hrs > 0 # ถ้ามีชั่วโมงปฏิบัติ ถือเป็น Lab
            return (total if total > 0 else 2), is_lab
    except: return 2, False
    return 2, False

# --- 4. Initialization (Morning Packer) ---
def create_compact_individual(courses, room_count, allowed_teachers_map):
    ind = [None] * len(courses)
    
    # Grouping
    groups = {}
    for idx, course in enumerate(courses):
        group_key = f"{course.get('department')}_{course.get('year_level')}_{course.get('group_no','')}"
        if group_key not in groups: groups[group_key] = []
        groups[group_key].append(idx)
    
    for g_key, course_indices in groups.items():
        # เริ่มต้นวางแผนตั้งแต่วันจันทร์ (Day 0)
        current_day = 0 
        current_slot = 0 # บังคับเริ่ม 08:30 เสมอ!
        
        # สุ่มลำดับวิชาในกลุ่มนิดหน่อย แต่ยังคง Concept เรียงเช้า
        random.shuffle(course_indices)
        
        for c_idx in course_indices:
            course = courses[c_idx]
            duration, is_lab = get_course_info(course)
            
            # สุ่มห้องและครูไปก่อน
            r = random.randint(0, room_count - 1)
            valid_teachers = allowed_teachers_map.get(c_idx, [0])
            ins = random.choice(valid_teachers) if valid_teachers else 0
            
            placed = False
            # พยายามวางลงใน slot ว่างถัดไปทันที
            for attempt in range(DAYS * 2): # ลองวนหาที่ลง
                
                # ถ้า slot ปัจจุบัน + duration มันทะลุเที่ยง (เช่นเริ่ม 10:30 เรียน 3 ชม.)
                # ให้กระโดดข้ามพักเที่ยงไปเริ่มบ่าย (Slot 5)
                if current_slot < LUNCH_SLOT and (current_slot + duration) > LUNCH_SLOT:
                    current_slot = 5 
                
                # ถ้าเกินเวลาเลิกเรียน (Slot 9 คือ 17:30)
                if (current_slot + duration) > 9:
                    current_day = (current_day + 1) % DAYS # ขึ้นวันใหม่
                    current_slot = 0 # เริ่มเช้าใหม่ 08:30
                    continue

                final_slot = (current_day * SLOTS_PER_DAY) + current_slot
                ind[c_idx] = [r, final_slot, ins]
                
                current_slot += duration # ขยับเวลาไปต่อท้าย
                placed = True
                break
            
            # กันเหนียว ถ้าวนหาไม่ได้จริงๆ (วิชาล้น)
            if not placed:
                d = random.randint(0, DAYS-1)
                s = random.choice([0, 1, 5])
                ind[c_idx] = [r, (d * SLOTS_PER_DAY) + s, ins]

    return creator.Individual(ind)

# --- 5. Mutation (Morning Gravity) ---
def gravity_mutate(individual, room_count, total_slots, allowed_teachers_map, courses, indpb=0.2):
    for i, gene in enumerate(individual):
        # Mutate Room
        if random.random() < indpb: 
            gene[0] = random.randint(0, room_count - 1)
        
        # Mutate Time (ดึงกลับมาเช้า)
        if random.random() < indpb:
            day = random.randint(0, DAYS - 1)
            # เพิ่มโอกาสลง Slot 0 (08:30) ให้สูงมากๆ
            slot_weights = [0, 0, 0, 1, 2, 5, 5, 6] 
            slot = random.choice(slot_weights)
            gene[1] = day * SLOTS_PER_DAY + slot
        
        # Mutate Instructor
        if random.random() < indpb:
            valid = allowed_teachers_map.get(i, [])
            if valid: gene[2] = random.choice(valid)
            
        # Correction Boundary
        duration, _ = get_course_info(courses[i])
        day = gene[1] // SLOTS_PER_DAY
        slot = gene[1] % SLOTS_PER_DAY
        
        # ดันไม่ให้ทับเที่ยง
        if slot < LUNCH_SLOT and (slot + duration) > LUNCH_SLOT:
            gene[1] = (day * SLOTS_PER_DAY) + 5
        # ดันไม่ให้เกินเย็น
        if slot + duration > 9:
            gene[1] -= (slot + duration - 9)

    return individual,

# --- 6. Fitness Function (Rule Enforcer) ---
def evaluate(individual, courses, room_ids, instructor_ids, room_details):
    penalty = 0
    room_usage = {}
    instructor_usage = {}
    group_usage = {}
    group_timelines = {} 

    for i, gene in enumerate(individual):
        r_idx, start_slot, i_idx = gene
        
        if r_idx >= len(room_ids) or i_idx >= len(instructor_ids):
            penalty += 50000
            continue

        room_code = room_ids[r_idx]
        instructor_id = instructor_ids[i_idx]
        course = courses[i]
        
        group_id = f"{course.get('department')}_{course.get('year_level')}_{course.get('group_no','')}"
        duration, is_lab = get_course_info(course)
        
        day_of_week = start_slot // SLOTS_PER_DAY
        slot_in_day = start_slot % SLOTS_PER_DAY
        end_slot = slot_in_day + duration

        # --- RULE 1: พักเที่ยงศักดิ์สิทธิ์ ---
        # ช่วงเรียนห้ามคาบเกี่ยว Slot 4 (12:30-13:30)
        if LUNCH_SLOT in range(slot_in_day, end_slot): 
            penalty += 1_000_000 

        # --- RULE 2: ประเภทห้อง (Room Matching) ---
        current_r_type = room_details.get(room_code, '').lower()
        is_room_lab = any(x in current_r_type for x in ['lab', 'shop', 'ปฏิบัติ', 'โรงฝึก', 'คอม'])
        
        if is_lab and not is_room_lab:
            penalty += 50_000 # วิชา Lab ไปลงห้องทฤษฎี -> ผิด
        elif not is_lab and is_room_lab:
            penalty += 5_000  # วิชาทฤษฎีไปแย่งห้อง Lab -> ผิดนิดหน่อย (แต่ยอมได้ถ้าห้องเต็ม)

        # --- RULE 3: ห้ามเรียนดึก ---
        if end_slot > 9: # เลิกหลัง 17:30
            penalty += 100_000 

        # --- RULE 4: การชนกัน (Conflicts) ---
        for t in range(duration):
            curr = start_slot + t
            
            if (curr, room_code) in room_usage: penalty += 500_000
            else: room_usage[(curr, room_code)] = True
            
            if (curr, instructor_id) in instructor_usage: penalty += 500_000
            else: instructor_usage[(curr, instructor_id)] = True
            
            if (curr, group_id) in group_usage: penalty += 500_000
            else: group_usage[(curr, group_id)] = True

        # เก็บข้อมูลไปวิเคราะห์รายวัน
        if group_id not in group_timelines: group_timelines[group_id] = {}
        if day_of_week not in group_timelines[group_id]: group_timelines[group_id][day_of_week] = []
        group_timelines[group_id][day_of_week].append((slot_in_day, end_slot))

    # --- Analyze Group Timelines ---
    for group_id, days_data in group_timelines.items():
        for day, classes in days_data.items():
            classes.sort(key=lambda x: x[0])
            
            # --- RULE 5: บังคับเริ่ม 08:30 (Student Morning First) ---
            first_start = classes[0][0]
            if first_start > 0: 
                # ถ้าคาบแรกไม่ใช่ 08:30 (Slot 0) โดนปรับหนักมาก!
                # ยกเว้นถ้าเป็นคาบที่ต่อจากพักเที่ยง (เริ่ม 13:30) อนุโลมได้นิดหน่อย
                if first_start == 5: 
                    penalty += 50_000 # เริ่มบ่ายเลย โดยเช้าว่าง (ไม่ดี)
                else:
                    penalty += 200_000 # เริ่มสาย (09:30, 10:30) ผิดมหันต์

            # --- RULE 6: ห้ามมีช่องว่าง (Gap) ---
            for k in range(len(classes) - 1):
                curr_end = classes[k][1]
                next_start = classes[k+1][0]
                gap = next_start - curr_end
                
                if gap > 0:
                    # อนุโลม gap ที่เป็นพักเที่ยง (จบ 4 เริ่ม 5)
                    if curr_end == 4 and next_start == 5:
                        pass 
                    else:
                        penalty += (gap * 200_000) # ช่องว่างระหว่างวิชา โดนหนัก

    return (penalty,)

# --- 7. Main Process ---
def run_genetic_algorithm(mode='balanced'):
    print(f"🧬 AI SCHEDULER STARTED... MODE: {mode.upper()}")
    
    cfg = GEN_CONFIGS.get(mode, GEN_CONFIGS['balanced'])

    try:
        # Load Data
        courses = supabase.table('curriculums').select("*, subjects(*)").execute().data
        rooms = supabase.table('classrooms').select("*").execute().data
        instructors = supabase.table('instructors').select("*").execute().data
        
        if not courses or not rooms or not instructors:
            return {"status": "error", "message": "Data incomplete"}

        room_ids = [r['room_code'] for r in rooms]
        instructor_ids = [i['id'] for i in instructors]
        
        instructor_name_map = {
            (ins['first_name'].strip(), ins['last_name'].strip()): int(ins['id']) 
            for ins in instructors
        }
        instructor_db_id_to_index = {int(ins['id']): idx for idx, ins in enumerate(instructors)}
        room_details = {r['room_code']: r.get('room_type', '') for r in rooms}

        # Allowed Teachers
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
                            rid = instructor_name_map[key]
                            if rid in instructor_db_id_to_index:
                                valid_indices.append(instructor_db_id_to_index[rid])
            
            if not valid_indices: valid_indices = list(range(len(instructors)))
            allowed_teachers_map[idx] = valid_indices

        # Register Toolbox
        for alias in ['individual', 'population', 'evaluate', 'mutate', 'mate', 'select']:
            if hasattr(toolbox, alias): toolbox.unregister(alias)

        toolbox.register("individual", create_compact_individual, 
                        courses=courses, 
                        room_count=len(room_ids),
                        allowed_teachers_map=allowed_teachers_map)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", gravity_mutate, 
                        room_count=len(room_ids), 
                        total_slots=TOTAL_SLOTS, 
                        allowed_teachers_map=allowed_teachers_map,
                        courses=courses,
                        indpb=cfg['mutation_prob']) 
        toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("evaluate", evaluate, 
                        courses=courses, 
                        room_ids=room_ids, 
                        instructor_ids=instructor_ids,
                        room_details=room_details)

        # Evolution Loop
        best_overall = None
        best_overall_fitness = float('inf')

        for run_idx in range(cfg['runs']):
            print(f"   🔄 Run {run_idx+1}/{cfg['runs']}")
            
            pop = toolbox.population(n=cfg['pop_size'])
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            
            pop, log = algorithms.eaSimple(
                pop, toolbox, 
                cxpb=0.8,    
                mutpb=cfg['mutation_prob'],   
                ngen=cfg['generations'],
                stats=stats, 
                halloffame=hof, 
                verbose=False
            )

            current_best = hof[0]
            fit = current_best.fitness.values[0]
            print(f"      ✅ Score: {fit:,.0f}")

            if fit < best_overall_fitness:
                best_overall = current_best
                best_overall_fitness = fit

        print(f"🏆 FINAL BEST FITNESS: {best_overall_fitness}")
        save_to_db(best_overall, courses, room_ids, instructor_ids)
        return {"status": "success", "mode": mode, "penalty": best_overall_fitness}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def save_to_db(best_schedule, courses, room_ids, instructor_ids):
    print("💾 Saving to database...")
    supabase.table('generated_schedules').delete().neq('id', 0).execute()
    data_list = []
    
    for i, gene in enumerate(best_schedule):
        r, start_slot, ins = gene
        course = courses[i]
        duration, _ = get_course_info(course)
        
        s_name = "Unknown"
        subj = course.get('subjects')
        if isinstance(subj, list) and subj: subj = subj[0]
        if isinstance(subj, dict): s_name = subj.get('subject_name', 'Unknown')
        
        for t in range(duration):
            current_slot = start_slot + t
            day = current_slot // SLOTS_PER_DAY
            slot = current_slot % SLOTS_PER_DAY
            
            if day != (start_slot // SLOTS_PER_DAY): continue

            data_list.append({
                "subject_code": course.get('subject_code', 'N/A'),
                "subject_name": s_name,
                "room_code": room_ids[r],
                "instructor_id": int(instructor_ids[ins]),
                "day_of_week": int(day),
                "start_slot": int(slot),
                "department": course.get('department', 'General'),
                "year_level": course.get('year_level', 'N/A')
            })
    
    batch_size = 500
    for k in range(0, len(data_list), batch_size):
        supabase.table('generated_schedules').insert(data_list[k:k+batch_size]).execute()
    print("✅ Saved successfully!")