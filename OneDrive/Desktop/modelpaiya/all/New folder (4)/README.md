# Career Prediction System - University of Kelaniya

A web-based career prediction system for Computer Science students at the University of Kelaniya. The system predicts potential career outcomes and provides personalized recommendations based on academic performance, skills, and experience.

## Features

- User-friendly web interface for data input
- Predicts starting salary with confidence intervals
- Provides personalized career insights
- Generates actionable recommendations
- Handles both required and optional student information
- Automatic estimation of missing data points

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

## Required Input Fields

- Student ID
- Gender
- Age at Enrollment (17-30)
- Province
- District
- A/L Z-score (1.0-2.5)
- Academic Pathway
- Intake Year
- Current Semester (1-8)

## Optional Input Fields

- Current GPA (0.0-4.0)
- Completed Internships
- Total Internship Months
- Completed Projects
- Professional Certifications

## System Requirements

- Python 3.8 or higher
- Modern web browser
- Internet connection (for Bootstrap CDN)

## Notes

- The system will automatically estimate missing optional information based on available data
- Providing more optional information will improve prediction accuracy
- All predictions include confidence intervals to indicate reliability
- Recommendations are tailored to the student's current academic stage

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 