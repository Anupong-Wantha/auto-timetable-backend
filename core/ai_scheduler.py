import random
import traceback
import numpy as np
from deap import base, creator, tools, algorithms
from core.database import supabase

# --- 1. Setup DEAP (เหมือนเดิม) ---
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# --- 2. Constants & Config (ปรับตามกฎข้อ 8, 10) ---
# 08:00 - 17:00 (9 ชั่วโมงเรียน + 1 พักเที่ยง = 10 Slots)
# Slot 0=08:00, 1=09:00, 2=10:00, 3=11:00, 4=12:00(Lunch), 5=13:00 ...
DAYS = 5
SLOTS_PER_DAY = 10  
TOTAL_SLOTS = DAYS * SLOTS_PER_DAY
LUNCH_SLOT = 4  # 12:00 - 13:00 (กฎข้อ 10)

# Config Presets
GEN_CONFIGS = {
    'balanced': {'pop_size': 500, 'generations': 150, 'runs': 1, 'mutation_prob': 0.5}
}

# --- 3. Helper Functions ---
def get_course_metadata(course):
    """ดึงข้อมูลเชิงลึกเพื่อใช้ตรวจสอบกฎ"""
    subj = course.get('subjects', {}) or {}
    if isinstance(subj, list): subj = subj[0]
    
    # Duration
    t_hrs = int(subj.get('theory_hours') or 0)
    p_hrs = int(subj.get('practice_hours') or 0)
    total_hours = t_hrs + p_hrs
    duration = total_hours if total_hours > 0 else 1
    
    # Metadata
    subj_name = subj.get('subject_name', '').lower()
    subj_code = course.get('subject_code', '').lower()
    
    # Flags
    is_scout = 'ลูกเสือ' in subj_name or 'scout' in subj_name  # กฎข้อ 7
    is_computer_subj = 'คอมพิวเตอร์' in subj_name or 'computer' in subj_name or 'code' in subj_code # กฎข้อ 16
    
    # Mock Advisor (สมมติว่าใน DB มี field นี้สำหรับกฎข้อ 17)
    # ถ้าไม่มีให้ถือว่า instructor ใน course คือ advisor
    advisor_id = course.get('advisor_id') 
    
    return duration, is_scout, is_computer_subj, advisor_id

def is_head_of_department(instructor_id):
    # TODO: เชื่อมข้อมูลจริงจาก DB ว่าใครเป็นหัวหน้า
    # ตัวอย่าง: return instructor_id == 101
    return False 

def is_computer_teacher(instructor_id, instructor_dept):
    # กฎข้อ 12
    return 'คอม' in instructor_dept or 'computer' in instructor_dept

# --- 4. Initialization (ปรับให้รองรับกฎข้อ 7 ลูกเสือ) ---
def create_smart_individual(courses, room_count, allowed_teachers_map, room_ids):
    ind = [None] * len(courses)
    
    stadium_idx = -1
    # ค้นหา index ของสนาม (Stadium) สำหรับลูกเสือ
    for idx, r_code in enumerate(room_ids):
        if 'สนาม' in r_code or 'stadium' in r_code.lower():
            stadium_idx = idx
            break
    if stadium_idx == -1: stadium_idx = 0 # Fallback

    for i, course in enumerate(courses):
        duration, is_scout, _, _ = get_course_metadata(course)
        
        # กฎข้อ 7: ลูกเสือ Fix เวลาและสถานที่
        if is_scout:
            # วันพุธ (Day 2) เวลา 15:00 (Slot 7)
            # Slot: 0=8, 1=9, 2=10, 3=11, 4=12(Lunch), 5=13, 6=14, 7=15
            scout_slot = (2 * SLOTS_PER_DAY) + 7 
            
            valid_teachers = allowed_teachers_map.get(i, [0])
            teacher = random.choice(valid_teachers) if valid_teachers else 0
            
            ind[i] = [stadium_idx, scout_slot, teacher]
            continue

        # วิชาทั่วไป: สุ่มแบบปกติ (Morning Packer Logic)
        r = random.randint(0, room_count - 1)
        valid_teachers = allowed_teachers_map.get(i, [0])
        ins = random.choice(valid_teachers) if valid_teachers else 0
        
        # สุ่ม Slot (พยายามเลี่ยงเที่ยง)
        d = random.randint(0, DAYS-1)
        s = random.randint(0, 8) 
        if s == LUNCH_SLOT: s = 5 # ถ้าสุ่มโดนเที่ยง ปัดเป็นบ่าย
        
        final_slot = (d * SLOTS_PER_DAY) + s
        ind[i] = [r, final_slot, ins]

    return creator.Individual(ind)

# --- 5. Mutation (Standard) ---
# (ใช้ logic เดิมได้ แต่เพิ่มการเช็คว่าถ้าเป็น gene ลูกเสือ ห้าม mutate เวลา/ห้อง)
def smart_mutate(individual, courses, room_count, allowed_teachers_map, indpb=0.2):
    for i, gene in enumerate(individual):
        _, is_scout, _, _ = get_course_metadata(courses[i])
        
        if is_scout: continue # กฎข้อ 7: ห้ามแก้ลูกเสือ
        
        if random.random() < indpb: # Mutate Room
            gene[0] = random.randint(0, room_count - 1)
        
        if random.random() < indpb: # Mutate Time
            d = random.randint(0, DAYS - 1)
            s = random.choice([0, 1, 2, 3, 5, 6, 7]) # Weight ลงเช้า/บ่าย เลี่ยงเที่ยง
            gene[1] = (d * SLOTS_PER_DAY) + s
            
        if random.random() < indpb: # Mutate Teacher
            valid = allowed_teachers_map.get(i, [])
            if valid: gene[2] = random.choice(valid)
    return individual,

# --- 6. Fitness Function (Rule Enforcer: 17 Rules) ---
def evaluate(individual, courses, room_ids, instructor_ids, room_details, instructor_details):
    penalty = 0
    
    # Tracking Dictionaries
    room_usage = {}
    teacher_usage = {}      # (slot, teacher_id) -> count
    student_usage = {}      # (slot, group_id) -> count
    
    teacher_hours = {tid: 0 for tid in instructor_ids}
    teacher_days_active = {tid: set() for tid in instructor_ids} # เก็บวันที่สอน
    
    # Cache ID maps
    id_to_teacher_obj = {str(ins['id']): ins for ins in instructor_details}

    for i, gene in enumerate(individual):
        r_idx, start_slot, i_idx = gene
        
        # Decode Gene
        room_code = room_ids[r_idx]
        teacher_id = instructor_ids[i_idx] # DB ID
        teacher_obj = id_to_teacher_obj.get(str(teacher_id), {})
        teacher_dept = teacher_obj.get('department', '')
        
        course = courses[i]
        group_id = f"{course.get('department')}_{course.get('year_level')}_{course.get('group_no','')}"
        
        duration, is_scout, is_comp_subj, advisor_id = get_course_metadata(course)
        
        day = start_slot // SLOTS_PER_DAY
        slot = start_slot % SLOTS_PER_DAY
        end_slot = slot + duration

        # --- Basic Constraints ---
        
        # กฎข้อ 10: พักเที่ยง 12:00-13:00 (Slot 4) ห้ามเรียน
        # ถ้าช่วงเวลาเรียนคาบเกี่ยว Slot 4
        if LUNCH_SLOT in range(slot, end_slot):
            penalty += 1_000_000

        # กฎข้อ 8: เริ่ม 08.00 (Slot 0) - Code บังคับ Slot 0 เป็น 08:00 อยู่แล้ว
        
        # กฎข้อ 9: ไม่เรียนเกิน 17:00 (Slot 9)
        if end_slot > 9:
            penalty += 100_000 # ปรับหนักถ้าเลยเวลา

        # กฎข้อ 16: วิชาคอม ต้องเรียนห้องคอม
        is_room_comp = 'คอม' in room_details.get(room_code, '') or 'computer' in room_details.get(room_code, '').lower()
        if is_comp_subj and not is_room_comp:
            penalty += 50_000
        elif not is_comp_subj and is_room_comp:
            penalty += 5_000 # วิชาอื่นมาใช้ห้องคอม (กันห้องเต็ม)

        # กฎข้อ 7 & 17: ลูกเสือ
        if is_scout:
            # เช็คเวลา (ต้องวันพุธ 15:00-17:00 คือ Day 2, Slot 7-9)
            if day != 2 or slot != 7:
                penalty += 500_000
            # เช็คสถานที่ (ต้องสนาม)
            if 'สนาม' not in room_code and 'stadium' not in room_code.lower():
                penalty += 100_000
            # กฎข้อ 17: ครูต้องเป็นที่ปรึกษา
            if advisor_id and int(advisor_id) != int(teacher_id):
                penalty += 200_000

        # กฎข้อ 13: ครูเมธา ว่าง จ.เช้า / ศ.บ่าย
        # สมมติ ID ครูเมธา = 999 หรือเช็คชื่อ
        f_name = teacher_obj.get('first_name', '')
        if 'เมธา' in f_name:
            # จันทร์ (Day 0) 08:00-12:00 (Slot 0-4)
            if day == 0 and slot < 4: penalty += 20_000
            # ศุกร์ (Day 4) 13:00-16:00 (Slot 5-8)
            if day == 4 and slot >= 5: penalty += 20_000

        # --- Time Loop Checks (Collision & Load) ---
        for t in range(duration):
            curr_abs = start_slot + t
            curr_slot_in_day = slot + t
            
            # Record Load
            teacher_hours[teacher_id] += 1
            teacher_days_active[teacher_id].add(day)

            # กฎข้อ 6: ห้องชนกัน
            if (curr_abs, room_code) in room_usage: penalty += 500_000
            else: room_usage[(curr_abs, room_code)] = True

            # กฎข้อ 4, 5: ครูชนกัน (สอน > 1 วิชา หรือ > 1 กลุ่ม)
            # Logic: ถ้าครูคนนี้ถูกจองในเวลานี้แล้ว = ชน
            if (curr_abs, teacher_id) in teacher_usage: penalty += 500_000
            else: teacher_usage[(curr_abs, teacher_id)] = True

            # กฎข้อ 3: นักเรียนกลุ่มเดียวกัน เรียนซ้อนกัน
            if (curr_abs, group_id) in student_usage: penalty += 500_000
            else: student_usage[(curr_abs, group_id)] = True

    # --- Aggregate Checks (ตรวจสอบภาพรวมหลังจัดตารางเสร็จ) ---
    
    hours_list = []
    
    for tid in instructor_ids:
        h = teacher_hours[tid]
        teacher_obj = id_to_teacher_obj.get(str(tid), {})
        is_head = is_head_of_department(tid) # Function check ID
        
        # กฎข้อ 1: หัวหน้าสอน 18-24 ชม.
        if is_head:
            if h < 18 or h > 24: penalty += 50_000
            
        # กฎข้อ 2: ครูทุกคนต้องสอนอย่างน้อย 18 ชม.
        elif h < 18: 
            penalty += 10_000 * (18 - h) # ปรับตามจำนวนชั่วโมงที่ขาด

        # กฎข้อ 12: ครูคอมพิวเตอร์ ต้องสอนทุกวัน (Active 5 วัน)
        if is_computer_teacher(tid, teacher_obj.get('department', '')):
            if len(teacher_days_active[tid]) < 5:
                penalty += 20_000 # ไม่มาสอนทุกวัน

        if h > 0: hours_list.append(h)

    # กฎข้อ 14: เกลี่ยชั่วโมงสอนให้ใกล้เคียงกัน (Standard Deviation)
    if hours_list:
        std_dev = np.std(hours_list)
        penalty += (std_dev * 1000) # ยิ่งกระจายมาก ยิ่งโดนปรับ

    # กฎข้อ 11: ไม่ฉีกคาบเรียน (Contiguous)
    # ถูกจัดการโดย Gene Structure แล้ว (1 ยีน = 1 ก้อนเวลาต่อเนื่อง) 
    # ดังนั้น penalty ส่วนนี้เป็น 0 โดยธรรมชาติ

    return (penalty,)

# --- 7. Main Execution (ส่วนที่ขาดหายไป) ---
def run_genetic_algorithm(mode='balanced'):
    print(f"🧬 AI SCHEDULER STARTED... MODE: {mode.upper()}")
    
    cfg = GEN_CONFIGS.get(mode, GEN_CONFIGS['balanced'])

    try:
        # 1. Load Data
        courses = supabase.table('curriculums').select("*, subjects(*)").execute().data
        rooms = supabase.table('classrooms').select("*").execute().data
        instructors = supabase.table('instructors').select("*").execute().data
        
        if not courses or not rooms or not instructors:
            return {"status": "error", "message": "Data incomplete"}

        # 2. Prepare Maps & IDs
        room_ids = [r['room_code'] for r in rooms]
        instructor_ids = [i['id'] for i in instructors]
        room_details = {r['room_code']: r.get('room_type', '') for r in rooms}
        
        # สร้าง Map สำหรับตรวจสอบชื่อครู (ใช้ตอน match วิชา)
        instructor_name_map = {
            (ins['first_name'].strip(), ins['last_name'].strip()): int(ins['id']) 
            for ins in instructors
        }
        instructor_db_id_to_index = {int(ins['id']): idx for idx, ins in enumerate(instructors)}

        # 3. Create Allowed Teachers Map (จับคู่ครูที่สอนได้ในแต่ละวิชา)
        allowed_teachers_map = {} 
        for idx, course in enumerate(courses):
            valid_indices = []
            subj_data = course.get('subjects')
            if isinstance(subj_data, list) and subj_data: subj_data = subj_data[0]
            
            if subj_data:
                # เช็ค slot ครูคนที่ 1-5 ในวิชา
                for k in range(1, 6): 
                    fname = subj_data.get(f'instructor_{k}_fname')
                    lname = subj_data.get(f'instructor_{k}_lname')
                    if fname and lname:
                        key = (fname.strip(), lname.strip())
                        if key in instructor_name_map:
                            rid = instructor_name_map[key]
                            if rid in instructor_db_id_to_index:
                                valid_indices.append(instructor_db_id_to_index[rid])
            
            # ถ้าไม่ระบุครู ให้สุ่มใครก็ได้ไปก่อน (หรือจะจัดการ error ทีหลัง)
            if not valid_indices: valid_indices = list(range(len(instructors)))
            allowed_teachers_map[idx] = valid_indices

        # 4. Register Toolbox (สำคัญมาก: อัปเดตพารามิเตอร์ให้ตรงกับฟังก์ชันใหม่)
        # Reset การลงทะเบียนเก่า
        for alias in ['individual', 'population', 'evaluate', 'mutate', 'mate', 'select']:
            if hasattr(toolbox, alias): toolbox.unregister(alias)

        # Register: Individual (ต้องส่ง room_ids ไปเช็คสนามบอลด้วย)
        toolbox.register("individual", create_smart_individual, 
                         courses=courses, 
                         room_count=len(room_ids),
                         allowed_teachers_map=allowed_teachers_map,
                         room_ids=room_ids)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        
        # Register: Mutate (ต้องส่ง courses ไปเช็คว่าเป็นวิชาลูกเสือหรือไม่)
        toolbox.register("mutate", smart_mutate, 
                         courses=courses,
                         room_count=len(room_ids), 
                         allowed_teachers_map=allowed_teachers_map,
                         indpb=cfg['mutation_prob']) 
        
        toolbox.register("select", tools.selTournament, tournsize=5)
        
        # Register: Evaluate (สำคัญ: ต้องส่ง instructor_details ไปเช็คกฎครู)
        toolbox.register("evaluate", evaluate, 
                         courses=courses, 
                         room_ids=room_ids, 
                         instructor_ids=instructor_ids,
                         room_details=room_details,
                         instructor_details=instructors)

        # 5. Evolution Loop
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
        
        # 6. Save Result
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
            r, start_slot, ins = gene
            course = courses[i]
            
            # ใช้ฟังก์ชัน get_course_metadata ที่เขียนไว้ด้านบน
            duration, _, _, _ = get_course_metadata(course)
            
            s_name = "Unknown"
            subj = course.get('subjects')
            if isinstance(subj, list) and subj: subj = subj[0]
            if isinstance(subj, dict): s_name = subj.get('subject_name', 'Unknown')
            
            for t in range(duration):
                current_slot = start_slot + t
                day = current_slot // SLOTS_PER_DAY
                slot = current_slot % SLOTS_PER_DAY
                
                # ป้องกันข้ามวัน (แม้ logic จะกันไว้แล้ว)
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
        
        # Batch Insert
        batch_size = 500
        for k in range(0, len(data_list), batch_size):
            supabase.table('generated_schedules').insert(data_list[k:k+batch_size]).execute()
        print("✅ Saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving to DB: {e}")
        traceback.print_exc()