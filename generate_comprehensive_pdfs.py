import os
from fpdf import FPDF

out_dir = "test_documents_comprehensive"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'SIR C.R REDDY COLLEGE OF ENGINEERING', 0, 1, 'C')
        self.set_font('Arial', 'I', 12)
        self.cell(0, 10, 'Department of Computer Science and Engineering', 0, 1, 'C')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln()

docs = [
    {
        "filename": "C1_Vision_Mission_PEOs.pdf",
        "title": "Criterion 1: Vision, Mission, and Program Educational Objectives",
        "body": """1.1 Vision of the Institute
To emerge as a premier institution in the field of engineering education and research, producing globally competent professionals with strong ethical values.

1.2 Mission of the Institute
- To provide state-of-the-art infrastructure and conducive learning environment.
- To foster industry-institute interaction for practical exposure.
- To encourage innovation, research, and entrepreneurship.

1.3 Vision of the CSE Department
To be a center of excellence in Computer Science and Engineering, empowering students with advanced technical knowledge and problem-solving skills to meet industry demands.

1.4 Mission of the CSE Department
M1: To impart quality education through a well-designed curriculum and modern teaching methodologies.
M2: To promote research and development activities in emerging technologies like AI, IoT, and Cloud Computing.
M3: To instill professional ethics, leadership qualities, and lifelong learning attitude.

1.5 Program Educational Objectives (PEOs)
PEO1: Graduates will have successful careers in software industry, research, or academia.
PEO2: Graduates will apply their technical skills to solve complex engineering problems.
PEO3: Graduates will exhibit teamwork, communication skills, and ethical behavior in their professional lives.
"""
    },
    {
        "filename": "C2_Curriculum_Teaching_Learning.pdf",
        "title": "Criterion 2: Program Curriculum and Teaching-Learning Processes",
        "body": """2.1 Curriculum Design
The curriculum for the CSE program is designed based on the AICTE model curriculum and aligned with the university guidelines. It encompasses foundational courses, core engineering subjects, professional electives, and open electives. 
Recent additions to the curriculum include Artificial Intelligence, Data Science, and Blockchain Technology to bridge the industry-academia gap.

2.2 Teaching-Learning Process
The department employs a mix of traditional and modern teaching methodologies:
- Project-Based Learning (PBL): Implemented in courses like Software Engineering and Web Development.
- Flipped Classroom: Used for advanced programming courses.
- ICT Tools: Extensive use of smart boards, LMS (Moodle), and video lectures (NPTEL, Coursera).

2.3 Assessment Methods
Continuous Internal Evaluation (CIE) accounts for 30% of the total marks, comprising mid-term exams, assignments, and mini-projects. The Semester End Examination (SEE) accounts for 70%.

2.4 Industry Interaction
Guest lectures by industry experts are organized monthly. The department has MOUs with TCS, Infosys, and IBM for student internships and faculty training programs.
"""
    },
    {
        "filename": "C3_CO_PO_Mapping_Syllabus.pdf",
        "title": "Criterion 3: Course Outcomes and Program Outcomes (Sample Syllabus)",
        "body": """Course Code: CS401
Course Name: Artificial Intelligence
Credits: 3

Course Description:
This course introduces the fundamental concepts of Artificial Intelligence, including search algorithms, knowledge representation, machine learning, and neural networks.

Course Outcomes (COs):
CO1: Understand the basic principles and history of AI.
CO2: Apply problem-solving and search strategies to real-world scenarios.
CO3: Design expert systems using logic and knowledge representation.
CO4: Implement basic machine learning models for classification and regression.

CO-PO Mapping Matrix (1-Low, 2-Medium, 3-High):
CO1 -> PO1 (3), PO2 (2)
CO2 -> PO2 (3), PO3 (2), PO4 (1)
CO3 -> PO3 (3), PO4 (2), PO5 (2)
CO4 -> PO4 (3), PO5 (3), PO12 (2)

Program Specific Outcomes (PSOs):
PSO1: Ability to design and develop software applications using modern tools.
PSO2: Ability to apply AI and Data Science concepts to solve societal problems.
"""
    },
    {
        "filename": "C4_Student_Performance_Placement.pdf",
        "title": "Criterion 4: Students' Performance",
        "body": """4.1 Admissions
The CSE department has an intake of 180 students per year. For the academic year 2024-2025, 100% of the seats were filled. The average CET cutoff rank for admission has consistently improved over the last three years.

4.2 Academic Performance
Success Rate without backlogs:
- 2023-2024: 85%
- 2022-2023: 82%
- 2021-2022: 78%

4.3 Placement Details
The placement cell actively organizes campus recruitment drives. 
- 2023-2024: 150 out of 180 students placed. Highest Package: 18 LPA. Average Package: 5.5 LPA. Top Recruiters: TCS, Wipro, Infosys, Amazon.
- 2022-2023: 145 students placed. Highest Package: 15 LPA. Average Package: 4.8 LPA.

4.4 Higher Education
Approximately 10% of graduates pursue higher education (M.Tech/MS) in reputed institutions in India and abroad (e.g., IITs, NITs, Universities in USA and UK).

4.5 Entrepreneurship
3 student-led startups were incubated in the college's Innovation Cell during the last academic year.
"""
    },
    {
        "filename": "C5_Faculty_Contributions_Research.pdf",
        "title": "Criterion 5: Faculty Information and Contributions",
        "body": """5.1 Faculty Details
The CSE department has a total of 30 faculty members. 
- Professors: 4 (All with Ph.D.)
- Associate Professors: 8 (6 with Ph.D., 2 pursuing Ph.D.)
- Assistant Professors: 18 (4 pursuing Ph.D.)
Student-Faculty Ratio (SFR) is maintained at 15:1 as per AICTE norms.

5.2 Faculty Retention
The faculty retention rate for the last three years is 90%, indicating a highly stable and satisfied academic environment.

5.3 Research and Publications (2024-2025)
- Dr. A. Ramakrishna (HOD): 'Deep Learning approaches for predictive analytics in healthcare', IEEE Access.
- Dr. B. Sharma: 'Optimization of Cloud Resources using Genetic Algorithms', International Journal of Cloud Computing.
- Total Journal Publications: 25 (15 Scopus Indexed, 5 SCI Indexed).
- Total Conference Publications: 40.

5.4 Funded Projects and Grants
- Dr. A. Ramakrishna received an AICTE RPS Grant of Rs. 15 Lakhs for 'AI-based Smart Agriculture System' (2023-2026).
- Dr. C. Verma secured a DST SERB grant of Rs. 10 Lakhs for 'IoT based Traffic Management'.

5.5 Faculty Development Programs (FDPs)
Faculty members regularly attend FDPs. 20 faculty members completed ATAL FDPs in AI, ML, and Cyber Security.
"""
    },
    {
        "filename": "C6_Facilities_Lab_Infrastructure.pdf",
        "title": "Criterion 6: Facilities and Technical Support",
        "body": """6.1 Classrooms and Seminar Halls
The department has 8 spacious, well-ventilated classrooms equipped with LCD projectors, Wi-Fi, and smart boards. There is 1 dedicated seminar hall with a seating capacity of 150.

6.2 Laboratories
The CSE department maintains state-of-the-art laboratories:
- Programming Lab: 60 Intel Core i7 PCs, 16GB RAM. Used for C, C++, Java, and Python programming.
- Database Lab: 60 PCs with Oracle and MySQL servers.
- AI & Data Science Lab: 30 High-performance workstations with NVIDIA GPUs for deep learning projects.
- Hardware Lab: Microprocessor and Microcontroller kits (8086, Arduino, Raspberry Pi).

6.3 Software Tools
Licensed software includes Microsoft Campus Agreement, IBM Rational Rose, MATLAB, and various open-source tools (Eclipse, Android Studio, Jupyter).

6.4 Internet Bandwidth
The campus provides a dedicated 1 Gbps leased line Internet connection, ensuring seamless connectivity for students and faculty. Wi-Fi is available campus-wide.

6.5 Maintenance
A dedicated team of 3 system administrators ensures regular maintenance, software updates, and network security.
"""
    },
    {
        "filename": "C7_Continuous_Improvement_Feedback.pdf",
        "title": "Criterion 7: Continuous Improvement",
        "body": """7.1 Feedback System
The institution has a robust feedback mechanism to ensure continuous improvement:
- Student Feedback: Collected twice a semester for each course regarding teaching methodology, syllabus coverage, and clarity.
- Alumni Feedback: Collected annually during the Alumni Meet. Suggestions are used to update curriculum and training programs.
- Employer Feedback: Collected during placement drives to assess the employability skills of students.

7.2 Academic Audit
An Internal Quality Assurance Cell (IQAC) conducts academic audits at the end of every semester. The audit reviews course files, lesson plans, question paper quality, and CO-PO attainment levels.

7.3 Improvements based on CO-PO Attainment
If a Course Outcome (CO) does not meet the target attainment level, the faculty proposes corrective actions. For example, in CS305 (Operating Systems), target attainment for CO2 was not met in 2023. Corrective action: Additional tutorial classes and practical simulation exercises were introduced in 2024, resulting in improved attainment.

7.4 Infrastructure Upgrades
Based on industry trends and alumni feedback, the AI & Data Science lab was established with an investment of Rs. 25 Lakhs.
"""
    },
    {
        "filename": "C8_First_Year_Academics.pdf",
        "title": "Criterion 8: First Year Academics",
        "body": """8.1 First Year Faculty
The First Year department (Basic Sciences and Humanities) comprises highly qualified faculty in Mathematics, Physics, Chemistry, and English. The Student-Faculty Ratio for the first year is maintained at 20:1.

8.2 Induction Program
A mandatory 3-week induction program is conducted for freshers as per AICTE guidelines. It includes:
- Familiarization with the department and campus facilities.
- Lectures by eminent personalities.
- Proficiency modules in English and basic programming.
- Extracurricular activities and sports.

8.3 Academic Performance in First Year
The pass percentage of first-year students in the CSE branch has been consistently above 85% for the last three years. Remedial classes are conducted for slow learners identified after the first mid-term exams.

8.4 First Year Laboratories
Well-equipped labs for Engineering Physics, Engineering Chemistry, and Basic Programming ensure practical exposure from day one.
"""
    },
    {
        "filename": "C9_Student_Support_Systems.pdf",
        "title": "Criterion 9: Student Support Systems",
        "body": """9.1 Mentoring System
A strong mentoring system is in place. Each faculty member is assigned 15-20 students. Mentors meet their mentees fortnightly to discuss academic progress, career goals, and personal issues. A centralized digital portal tracks mentoring records.

9.2 Anti-Ragging and Grievance Redressal
The college maintains a strict zero-tolerance policy towards ragging. An active Anti-Ragging Committee and Squad monitor the campus. The Grievance Redressal Cell addresses student complaints efficiently.

9.3 Co-curricular and Extra-curricular Activities
The CSE department has active student chapters of IEEE, CSI, and ACM. 
- 'TechFest': An annual national-level technical symposium organizing coding contests, hackathons, and paper presentations.
- 'Cultural Fest': Annual cultural event promoting arts, music, and dance.
- Sports: Excellent sports facilities available; students actively participate in university-level tournaments.

9.4 Career Guidance and Training
The Training and Placement Cell conducts extensive training programs starting from the 2nd year. 
- 2nd Year: Communication skills and aptitude training.
- 3rd Year: Technical training (Data Structures, Algorithms) and coding bootcamps.
- 4th Year: Mock interviews, group discussions, and company-specific training.
"""
    },
    {
        "filename": "C10_Governance_Financial_Budget.pdf",
        "title": "Criterion 10: Governance, Institutional Support and Financial Resources",
        "body": """10.1 Governing Body
The Governing Body of Sir C.R. Reddy College of Engineering consists of visionary management representatives, eminent academicians, and industry experts. They meet twice a year to review institutional progress, approve budgets, and set strategic goals.

10.2 Delegation of Financial Powers
The Principal and HODs have delegated financial powers for routine operations, laboratory consumables, and organizing seminars/workshops, ensuring smooth academic functioning.

10.3 Financial Resources and Budget Allocation
The institution has adequate financial resources, primarily generated through student fees. 
Budget Allocation for CSE Department (2024-2025):
- Laboratory Equipment & Software: Rs. 30 Lakhs
- R&D and Faculty Development: Rs. 10 Lakhs
- Library and E-Resources: Rs. 5 Lakhs
- Maintenance and Consumables: Rs. 5 Lakhs

10.4 Library Facilities
The central library is fully automated and covers an area of 1500 sq. m. 
- Volumes: 50,000+
- Titles: 10,000+
- E-Journals: IEEE, Springer, ACM Digital Library subscriptions available.
The department also maintains a separate reference library for quick access by faculty and students.
"""
    }
]

for doc in docs:
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title(doc['title'])
    pdf.chapter_body(doc['body'])
    filepath = os.path.join(out_dir, doc['filename'])
    pdf.output(filepath)
    print(f"Generated {filepath}")
