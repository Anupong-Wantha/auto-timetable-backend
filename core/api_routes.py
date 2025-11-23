from flask import Blueprint, request
from flask_restx import Api, Resource, fields
from core.database import supabase
from core.ai_scheduler import run_genetic_algorithm

api_bp = Blueprint('api', __name__)

api = Api(api_bp, 
          version='1.1', 
          title='Timetable Scheduler API', 
          description='API สำหรับระบบจัดตารางสอน (Updated for Advanced Search)',
          doc='/docs')

# Namespaces
ns_stats = api.namespace('stats', description='สถิติข้อมูล')
ns_sched = api.namespace('schedules', description='ตารางเรียน')
ns_data = api.namespace('data', description='จัดการข้อมูล')
ns_ai = api.namespace('ai', description='ระบบ AI')

# ================= Models =================

student_model = api.model('Student', {
    'student_id': fields.String(required=True, description='รหัสนักศึกษา'),
    'first_name': fields.String(required=True, description='ชื่อจริง'),
    'last_name': fields.String(required=True, description='นามสกุล'),
    'department': fields.String(required=True, description='แผนกวิชา'),
    'year_level': fields.String(required=True, description='ชั้นปี'),
    'group_no': fields.String(description='กลุ่มที่ (เช่น 1, 2, A, B)')
})

instructor_model = api.model('Instructor', {
    'first_name': fields.String(required=True),
    'last_name': fields.String(required=True),
    'department': fields.String(required=True),
    'max_hours_per_week': fields.Integer(description='ชั่วโมงสอนสูงสุดต่อสัปดาห์')
})

classroom_model = api.model('Classroom', {
    'room_code': fields.String(required=True),
    'room_type': fields.String(),
    'capacity': fields.Integer(),
    'building': fields.String(),
    'department_owner': fields.String(description='แผนกที่เป็นเจ้าของห้อง')
})

subject_model = api.model('Subject', {
    'subject_code': fields.String(required=True),
    'subject_name': fields.String(required=True),
    'theory_hours': fields.Integer(),
    'practice_hours': fields.Integer(),
    'credits': fields.Integer(),
    'instructor_1_fname': fields.String(),
    'instructor_1_lname': fields.String(),
    'instructor_2_fname': fields.String(),
    'instructor_2_lname': fields.String(),
})

# ================= Stats & Basic Routes =================
@ns_stats.route('/')
class StatsResource(Resource):
    def get(self):
        try:
            count_std = supabase.table('students').select('*', count='exact').limit(0).execute().count
            count_ins = supabase.table('instructors').select('*', count='exact').limit(0).execute().count
            count_subj = supabase.table('subjects').select('*', count='exact').limit(0).execute().count
            count_rooms = supabase.table('classrooms').select('*', count='exact').limit(0).execute().count
            return {"students": count_std, "instructors": count_ins, "subjects": count_subj, "rooms": count_rooms}
        except Exception as e:
            print(f"🔥🔥 FIRE IN THE HOLE! Error fetching stats: {e}")  # ให้มันตะโกนออกมาใน Log
            return {"students": 0, "instructors": 0, "subjects": 0, "rooms": 0}



@ns_sched.route('/generate')
class GenerateAI(Resource):
    @api.doc(params={'mode': 'draft | balanced | perfect'}) # Documentation
    def post(self):
        try:
            # รับค่า JSON body
            data = request.json or {} 
            mode = data.get('mode', 'balanced') # ถ้าไม่ส่งมา ให้เป็น balanced
            
            return run_genetic_algorithm(mode=mode)
        except Exception as e:
            return {"error": str(e)}, 500

# ================= ADVANCED SEARCH (หัวใจหลักที่ปรับปรุง) =================

@ns_sched.route('/search')
class AdvancedSchedule(Resource):
    def get(self):
        """
        ค้นหาตารางเรียนแบบละเอียด รองรับ 4 โหมด: Student, Instructor, Room, Subject
        รองรับ Parameter จาก Frontend ใหม่ทั้งหมด
        """
        type_ = request.args.get('type') 
        query = supabase.table('generated_schedules').select('*')
        
        # Helper function to clean inputs
        def clean(val):
            return val.strip() if val else None

        try:
            # --- 1. SEARCH STUDENT ---
            if type_ == 'student':
                std_id = clean(request.args.get('id'))
                fname = clean(request.args.get('fname')) 
                lname = clean(request.args.get('lname'))
                dept = clean(request.args.get('dept'))
                year = clean(request.args.get('year'))
                group = clean(request.args.get('group'))
                
                # A. กรณีระบุตัวตนชัดเจน (ID หรือ ชื่อ) -> ไปค้นข้อมูลนักเรียนก่อนเพื่อเอา Dept/Year
                if std_id or fname or lname:
                    std_query = supabase.table('students').select('*')
                    if std_id: std_query = std_query.eq('student_id', std_id)
                    if fname: std_query = std_query.ilike('first_name', f'%{fname}%')
                    if lname: std_query = std_query.ilike('last_name', f'%{lname}%')
                    
                    students = std_query.execute().data
                    if not students:
                        return [] # ไม่เจอนักเรียน
                    
                    # เอาข้อมูลนักเรียนคนแรกที่เจอมาใช้เป็น Filter ตารางเรียน
                    # (สมมติว่าตารางเรียนจัดตาม แผนก และ ชั้นปี)
                    target = students[0]
                    query = query.eq('department', target['department']).eq('year_level', target['year_level'])
                
                # B. กรณีระบุแค่ filter (แผนก, ชั้นปี, กลุ่ม)
                else:
                    if dept: query = query.eq('department', dept)
                    if year: query = query.eq('year_level', year)
                    # หมายเหตุ: ถ้า DB generated_schedules ไม่ได้เก็บ group_no อาจต้องข้าม filter นี้
                    # หรือถ้ามีการเก็บในอนาคตให้ uncomment บรรทัดล่าง
                    # if group: query = query.eq('group_no', group)

            # --- 2. SEARCH INSTRUCTOR ---
            elif type_ == 'instructor':
                fname = clean(request.args.get('fname'))
                lname = clean(request.args.get('lname'))
                dept = clean(request.args.get('dept'))
                
                # ต้องหา ID ของอาจารย์ก่อน
                if fname or lname or dept:
                    ins_query = supabase.table('instructors').select('id')
                    if fname: ins_query = ins_query.ilike('first_name', f'%{fname}%')
                    if lname: ins_query = ins_query.ilike('last_name', f'%{lname}%')
                    if dept: ins_query = ins_query.eq('department', dept)
                    
                    instructors = ins_query.execute().data
                    if not instructors:
                        return []
                    
                    ins_ids = [str(i['id']) for i in instructors]
                    query = query.in_('instructor_id', ins_ids)

            # --- 3. SEARCH ROOM ---
            elif type_ == 'room':
                room_code = clean(request.args.get('room_code'))
                room_type = clean(request.args.get('room_type'))
                building = clean(request.args.get('building'))
                dept = clean(request.args.get('dept'))
                
                # ถ้าค้นหาด้วย code ตรงๆ
                if room_code:
                    query = query.ilike('room_code', f'%{room_code}%')
                
                # ถ้าค้นหาด้วยคุณสมบัติห้อง (ตึก, ประเภท, แผนกเจ้าของ) -> ต้องไปหา room_code จากตาราง classrooms
                if room_type or building or dept:
                    room_query = supabase.table('classrooms').select('room_code')
                    if room_type: room_query = room_query.eq('room_type', room_type)
                    if building: room_query = room_query.ilike('building', f'%{building}%')
                    if dept: room_query = room_query.eq('department_owner', dept)
                    
                    matching_rooms = room_query.execute().data
                    if not matching_rooms:
                        return [] # ไม่เจอห้องที่มีคุณสมบัตินี้
                        
                    valid_room_codes = [r['room_code'] for r in matching_rooms]
                    query = query.in_('room_code', valid_room_codes)

            # --- 4. SEARCH SUBJECT ---
            elif type_ == 'subject':
                code = clean(request.args.get('code'))
                name = clean(request.args.get('name'))
                instructor_name = clean(request.args.get('instructor'))
                
                if code: query = query.ilike('subject_code', f'%{code}%')
                if name: query = query.ilike('subject_name', f'%{name}%')
                
                # ถ้าค้นหาด้วยชื่อครูในหน้ารายวิชา -> ซับซ้อนหน่อย ต้องหาครู -> ได้ ID -> มาหาในตาราง
                if instructor_name:
                    ins_query = supabase.table('instructors').select('id') \
                        .or_(f"first_name.ilike.%{instructor_name}%,last_name.ilike.%{instructor_name}%")
                    
                    instructors = ins_query.execute().data
                    if instructors:
                        ins_ids = [str(i['id']) for i in instructors]
                        query = query.in_('instructor_id', ins_ids)
                    else:
                        return [] # ไม่เจอครูชื่อนี้

            # Execute Final Query
            # เรียงตาม วัน (0-4) และ เวลาเรียน (Slot)
            result = query.order('day_of_week').order('start_slot').execute().data
            return result

        except Exception as e:
            print(f"Search Error: {e}")
            return {"error": str(e)}, 400

# ================= Data Management (CRUD) =================

@ns_data.route('/students')
class ManageStudents(Resource):
    @api.expect(student_model)
    def post(self):
        try:
            res = supabase.table('students').insert(api.payload).execute()
            return res.data, 201
        except Exception as e:
            return {"error": str(e)}, 400

@ns_data.route('/instructors')
class ManageInstructors(Resource):
    @api.expect(instructor_model)
    def post(self):
        try:
            res = supabase.table('instructors').insert(api.payload).execute()
            return res.data, 201
        except Exception as e:
            return {"error": str(e)}, 400

@ns_data.route('/classrooms')
class ManageClassrooms(Resource):
    @api.expect(classroom_model)
    def post(self):
        try:
            res = supabase.table('classrooms').insert(api.payload).execute()
            return res.data, 201
        except Exception as e:
            return {"error": str(e)}, 400

@ns_data.route('/subjects')
class ManageSubjects(Resource):
    @api.expect(subject_model)
    def post(self):
        try:
            res = supabase.table('subjects').insert(api.payload).execute()
            return res.data, 201
        except Exception as e:
            return {"error": str(e)}, 400