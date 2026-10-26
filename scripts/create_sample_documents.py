"""Reproduce the three fictional demo PDFs: python scripts/create_sample_documents.py."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

DESTINATION = Path(__file__).resolve().parents[1] / "data" / "sample_documents"
POLICIES = {
    "Academic Regulations.pdf": [
        ("Academic Regulations", "Study structure and progression", [
            ("Scope and status", "Meridian Example University is fictional. This document is a fictional demonstration policy for CampusRAG, not an official policy of DTU or any real university. These example regulations apply to the fictional 2026-2027 academic year."),
            ("Semester structure", "The academic year has two teaching semesters. Each semester contains 16 teaching weeks followed by an examination period. Undergraduate and postgraduate students register for courses at the start of each semester."),
            ("Registration window", "Course registration for the fictional autumn semester opens on 3 August 2026 and closes on 14 August 2026. A student must meet the stated prerequisites before registering for a course."),
            ("Credit load", "B.Tech students normally register for 18 to 24 credits per semester. M.Tech students normally register for 12 to 18 credits per semester. A faculty adviser must approve any request outside the normal credit range."),
        ]),
        ("Academic Regulations", "Assessment and academic support", [
            ("Passing a course", "The minimum passing grade is D. Students must earn a cumulative grade point average of at least 5.0 out of 10 to graduate. A failed compulsory course must be repeated when it is next offered."),
            ("Advising", "Students with a semester grade point average below 5.0 must meet their faculty adviser before the next registration period. Academic support includes weekly tutoring and adviser office hours."),
            ("Policy boundaries", "Attendance eligibility is defined in the separate Attendance Guidelines. Examination conduct and re-evaluation procedures are defined in the separate Examination Guidelines. This document does not specify tuition fees, scholarship amounts, or hostel charges."),
        ]),
    ],
    "Attendance Guidelines.pdf": [
        ("Attendance Guidelines", "Minimum attendance and applicability", [
            ("Fictional example", "Meridian Example University is fictional. These are demonstration attendance rules for CampusRAG, not official DTU or other university policies. The example rules cover the fictional 2026-2027 academic year."),
            ("Minimum attendance", "Students must maintain at least 75% attendance in each course to be eligible for the end-semester examination. Attendance is calculated separately for each course; attendance from different courses cannot be combined."),
            ("Postgraduate applicability", "The same minimum attendance requirement of 75% in each course applies to M.Tech and other postgraduate students, as well as to B.Tech and other undergraduate students. There is no automatic postgraduate exemption."),
            ("Record review", "Students should review their attendance record every week. A suspected recording error must be reported to the course instructor within five working days of publication of the relevant attendance entry."),
        ]),
        ("Attendance Guidelines", "Absences and eligibility review", [
            ("Medical absences", "A student may request an attendance review for a documented medical absence by submitting a medical certificate to the department within seven calendar days of returning to classes. Submission does not automatically waive the 75% attendance requirement."),
            ("Laboratory participation", "Laboratory attendance is recorded separately from lecture attendance. A missed assessed laboratory activity must be discussed with the laboratory instructor; a replacement activity requires instructor approval."),
            ("Eligibility notification", "Departments publish provisional examination eligibility lists ten calendar days before the first end-semester examination. Students may request correction of factual attendance errors within three working days of publication."),
        ]),
    ],
    "Examination Guidelines.pdf": [
        ("Examination Guidelines", "Eligibility, assessment, and exam-day conduct", [
            ("Fictional example", "Meridian Example University is fictional. These examination guidelines are fictional CampusRAG examples and are not official policies of DTU or any real university. They apply to the example 2026-2027 academic year."),
            ("Examination eligibility", "Students must complete course registration and satisfy the attendance eligibility rules before taking an end-semester examination. The separate Attendance Guidelines state the attendance threshold."),
            ("Assessment weights", "For a standard theory course, internal assessment contributes 40% and the end-semester examination contributes 60% of the final course score. Practical courses may use a separately published assessment scheme."),
            ("Exam-day requirements", "Students must carry their university identity card and examination admit card. Students should arrive 30 minutes before the scheduled start. Mobile phones, smartwatches, and unauthorized notes are prohibited in the examination room."),
        ]),
        ("Examination Guidelines", "Results, re-evaluation, and missed examinations", [
            ("Re-evaluation deadline", "A request for re-evaluation must be submitted through the student portal within seven calendar days of publication of the examination result. The request must identify the course code and the assessment to be reviewed. This example document does not state a re-evaluation fee."),
            ("Missed examinations", "A student who misses an examination because of a documented emergency must contact the examination office within three working days. A make-up examination requires written approval; approval is not automatic."),
            ("Example examination dates", "For the fictional autumn 2026 semester, the end-semester examination period runs from 7 December 2026 to 18 December 2026. Individual course dates are published in the examination timetable."),
            ("Academic integrity", "Suspected misconduct is referred to the academic integrity committee. Students receive an opportunity to respond before a final decision. This document does not prescribe a fixed penalty for every type of misconduct."),
        ]),
    ],
}


def create_samples():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("PolicyTitle", fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=colors.HexColor("#132b42"), spaceAfter=12))
    styles.add(ParagraphStyle("PolicySubtitle", fontSize=13, leading=18, textColor=colors.HexColor("#087f8c"), spaceAfter=22))
    styles.add(ParagraphStyle("PolicyBody", fontSize=10.5, leading=16, spaceAfter=16, alignment=TA_LEFT))
    styles.add(ParagraphStyle("PolicyHeading", fontName="Helvetica-Bold", fontSize=12, leading=17, spaceAfter=6, textColor=colors.HexColor("#132b42")))

    def page_decoration(canvas, doc):
        width, height = A4
        canvas.setFillColor(colors.HexColor("#087f8c"))
        canvas.rect(0, height - 9 * mm, width, 9 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#586b7d"))
        canvas.drawString(22 * mm, 17 * mm, "FICTIONAL EXAMPLE | Meridian Example University | CampusRAG")
        canvas.drawRightString(width - 22 * mm, 17 * mm, f"Page {doc.page}")

    for filename, pages in POLICIES.items():
        flow = []
        for index, (title, subtitle, sections) in enumerate(pages):
            if index:
                flow.append(PageBreak())
            flow += [Paragraph("FICTIONAL UNIVERSITY POLICY / 2026-2027", styles["PolicySubtitle"]),
                     Paragraph(title, styles["PolicyTitle"]), Paragraph(subtitle, styles["PolicySubtitle"])]
            for heading, body in sections:
                flow += [Paragraph(heading, styles["PolicyHeading"]), Paragraph(body, styles["PolicyBody"])]
            flow.append(Spacer(1, 5 * mm))
        SimpleDocTemplate(str(DESTINATION / filename), pagesize=A4,
                          rightMargin=22 * mm, leftMargin=22 * mm,
                          topMargin=24 * mm, bottomMargin=26 * mm,
                          title=pages[0][0] + " - Fictional Example", author="CampusRAG").build(
                              flow, onFirstPage=page_decoration, onLaterPages=page_decoration)
        print(DESTINATION / filename)


if __name__ == "__main__":
    create_samples()
