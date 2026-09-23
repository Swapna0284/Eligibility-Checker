import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


class EligibilityChecker:

    def __init__(self):
        self.rules = self.define_eligibility_rules()
        self.programs = self.define_programs()
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []

    # --------------------------------------------------
    # Internship Programs
    # --------------------------------------------------
    def define_programs(self):
        return {
            "AI_ML_Internship": {
                "min_cgpa": 7.0,
                "min_year": 2,
                "required_skills": [
                    "Python",
                    "Machine Learning",
                    "Statistics"
                ],
                "max_backlogs": 2,
                "domains": [
                    "Computer Science",
                    "Data Science",
                    "AI"
                ]
            },

            "Data_Science_Internship": {
                "min_cgpa": 6.5,
                "min_year": 2,
                "required_skills": [
                    "Python",
                    "SQL",
                    "Data Visualization"
                ],
                "max_backlogs": 3,
                "domains": [
                    "Computer Science",
                    "Data Science",
                    "Statistics"
                ]
            },

            "Full_Stack_Dev_Internship": {
                "min_cgpa": 6.0,
                "min_year": 1,
                "required_skills": [
                    "Python",
                    "JavaScript",
                    "React",
                    "SQL"
                ],
                "max_backlogs": 4,
                "domains": [
                    "Computer Science",
                    "IT",
                    "Software Engineering"
                ]
            },

            "DevOps_Internship": {
                "min_cgpa": 6.0,
                "min_year": 2,
                "required_skills": [
                    "Linux",
                    "Docker",
                    "AWS",
                    "Python"
                ],
                "max_backlogs": 3,
                "domains": [
                    "Computer Science",
                    "IT",
                    "Networking"
                ]
            },

            "Cyber_Security_Internship": {
                "min_cgpa": 6.5,
                "min_year": 2,
                "required_skills": [
                    "Python",
                    "Network Security",
                    "Linux"
                ],
                "max_backlogs": 2,
                "domains": [
                    "Computer Science",
                    "Cyber Security",
                    "IT"
                ]
            }
        }

    # --------------------------------------------------
    # General Eligibility Rules
    # --------------------------------------------------
    def define_eligibility_rules(self):
        return {
            "minimum_cgpa": 6.0,
            "minimum_year": 2,
            "maximum_backlogs": 3,
            "minimum_skills": 2
        }

    # --------------------------------------------------
    # Rule-Based Eligibility
    # --------------------------------------------------
    def check_eligibility_rules(self, student_data):

        results = {
            "eligible": True,
            "checks": [],
            "messages": []
        }

        # CGPA
        cgpa_passed = student_data.get("cgpa", 0) >= 6.0

        results["checks"].append({
            "criterion": "CGPA",
            "passed": cgpa_passed
        })

        if not cgpa_passed:
            results["eligible"] = False
            results["messages"].append(
                "CGPA must be at least 6.0"
            )

        # Year
        year_passed = student_data.get("year_of_study", 0) >= 2

        results["checks"].append({
            "criterion": "Year of Study",
            "passed": year_passed
        })

        if not year_passed:
            results["eligible"] = False
            results["messages"].append(
                "Student must be in year 2 or above"
            )

        # Domain
        allowed_domains = [
            "Computer Science",
            "Data Science",
            "AI",
            "IT",
            "Software Engineering",
            "Cyber Security"
        ]

        domain_passed = student_data.get("domain", "") in allowed_domains

        results["checks"].append({
            "criterion": "Domain",
            "passed": domain_passed
        })

        if not domain_passed:
            results["eligible"] = False
            results["messages"].append(
                "Domain is not eligible"
            )

        # Backlogs
        backlog_passed = student_data.get("backlogs", 0) <= 3

        results["checks"].append({
            "criterion": "Backlogs",
            "passed": backlog_passed
        })

        if not backlog_passed:
            results["eligible"] = False
            results["messages"].append(
                "Maximum 3 backlogs are allowed"
            )

        # Skills
        skills = student_data.get("skills", [])
        skills_passed = len(skills) >= 2

        results["checks"].append({
            "criterion": "Skills",
            "passed": skills_passed
        })

        if not skills_passed:
            results["eligible"] = False
            results["messages"].append(
                "At least 2 relevant skills are required"
            )

        return results

    # --------------------------------------------------
    # Generate Training Data
    # --------------------------------------------------
    def prepare_training_data(self):

        np.random.seed(42)

        n_samples = 500

        domains = [
            "Computer Science",
            "Data Science",
            "AI",
            "IT",
            "Software Engineering",
            "Cyber Security"
        ]

        data = {
            "cgpa": np.random.uniform(5.0, 9.5, n_samples),
            "year_of_study": np.random.randint(1, 5, n_samples),
            "backlogs": np.random.randint(0, 5, n_samples),
            "domain": np.random.choice(domains, n_samples),
            "skills_count": np.random.randint(1, 6, n_samples),
            "has_experience": np.random.choice(
                [0, 1],
                n_samples,
                p=[0.6, 0.4]
            )
        }

        df = pd.DataFrame(data)

        df["eligible"] = (
            (df["cgpa"] >= 6.0)
            & (df["year_of_study"] >= 2)
            & (df["backlogs"] <= 3)
            & (df["skills_count"] >= 2)
            & df["domain"].isin(domains)
        ).astype(int)

        return df

    # --------------------------------------------------
    # Train Decision Tree
    # --------------------------------------------------
    def train_model(self):

        df = self.prepare_training_data()

        df_encoded = df.copy()

        self.label_encoders = {}

        for column in ["domain"]:

            encoder = LabelEncoder()

            df_encoded[column] = encoder.fit_transform(
                df_encoded[column]
            )

            self.label_encoders[column] = encoder

        self.feature_columns = [
            "cgpa",
            "year_of_study",
            "backlogs",
            "domain",
            "skills_count",
            "has_experience"
        ]

        X = df_encoded[self.feature_columns]
        y = df_encoded["eligible"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("\nModel Accuracy:", f"{accuracy:.2%}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nFeature Importance:")

        for feature, importance in zip(
            self.feature_columns,
            self.model.feature_importances_
        ):
            print(f"{feature}: {importance:.4f}")

        return accuracy

    # --------------------------------------------------
    # ML Prediction
    # --------------------------------------------------
    def predict_eligibility(self, student_data):

        if self.model is None:
            return {
                "error": "Model is not trained."
            }

        encoded_data = {}

        for column in self.feature_columns:

            if column in self.label_encoders:

                encoder = self.label_encoders[column]

                try:
                    encoded_data[column] = encoder.transform(
                        [student_data[column]]
                    )[0]

                except ValueError:
                    return {
                        "error": f"Unknown value for {column}"
                    }

            else:
                encoded_data[column] = student_data[column]

        features = np.array(
            [encoded_data[column] for column in self.feature_columns]
        ).reshape(1, -1)

        prediction = self.model.predict(features)[0]

        probability = self.model.predict_proba(features)[0]

        eligible_probability = (
            probability[1]
            if len(probability) > 1
            else probability[0]
        )

        if eligible_probability >= 0.70:
            confidence = "High"
        elif eligible_probability >= 0.50:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "eligible": bool(prediction),
            "probability": eligible_probability,
            "confidence": confidence
        }

    # --------------------------------------------------
    # Program Recommendations
    # --------------------------------------------------
    def get_program_recommendations(self, student_data):

        recommendations = {}

        for program, requirements in self.programs.items():

            eligible = True
            reasons = []

            if student_data.get("cgpa", 0) < requirements["min_cgpa"]:
                eligible = False
                reasons.append(
                    f"CGPA below {requirements['min_cgpa']}"
                )

            if student_data.get("year_of_study", 0) < requirements["min_year"]:
                eligible = False
                reasons.append(
                    f"Must be in year {requirements['min_year']} or above"
                )

            if student_data.get("domain", "") not in requirements["domains"]:
                eligible = False
                reasons.append(
                    "Domain not eligible for this program"
                )

            if student_data.get("backlogs", 0) > requirements["max_backlogs"]:
                eligible = False
                reasons.append(
                    f"Backlogs exceed {requirements['max_backlogs']}"
                )

            recommendations[program] = {
                "eligible": eligible,
                "reasons": reasons
            }

        return recommendations

    # --------------------------------------------------
    # Complete Report
    # --------------------------------------------------
    def generate_report(self, student_data):

        rule_result = self.check_eligibility_rules(
            student_data
        )

        ml_result = self.predict_eligibility(
            student_data
        )

        recommendations = self.get_program_recommendations(
            student_data
        )

        print("\n" + "=" * 60)
        print("AI INTERNSHIP ELIGIBILITY REPORT")
        print("=" * 60)

        print("\nStudent Profile")
        print("-" * 30)

        print("Name:", student_data.get("name"))
        print("CGPA:", student_data.get("cgpa"))
        print("Year:", student_data.get("year_of_study"))
        print("Domain:", student_data.get("domain"))
        print("Backlogs:", student_data.get("backlogs"))
        print("Skills:", ", ".join(student_data.get("skills", [])))

        print("\nRule-Based Result")
        print("-" * 30)

        if rule_result["eligible"]:
            print("Status: ELIGIBLE")
        else:
            print("Status: NOT ELIGIBLE")

        for check in rule_result["checks"]:

            status = "PASS" if check["passed"] else "FAIL"

            print(
                f"{check['criterion']}: {status}"
            )

        if rule_result["messages"]:

            print("\nIssues:")

            for message in rule_result["messages"]:
                print("-", message)

        print("\nML Prediction")
        print("-" * 30)

        if "error" not in ml_result:

            print(
                "Prediction:",
                "ELIGIBLE"
                if ml_result["eligible"]
                else "NOT ELIGIBLE"
            )

            print(
                "Probability:",
                f"{ml_result['probability']:.2%}"
            )

            print(
                "Confidence:",
                ml_result["confidence"]
            )

        print("\nProgram Recommendations")
        print("-" * 30)

        for program, result in recommendations.items():

            status = (
                "ELIGIBLE"
                if result["eligible"]
                else "NOT ELIGIBLE"
            )

            print(f"\n{program}: {status}")

            for reason in result["reasons"]:
                print("  -", reason)


# ======================================================
# Main Program
# ======================================================

if __name__ == "__main__":

    checker = EligibilityChecker()

    # Train ML model
    print("Training Decision Tree Model...")

    checker.train_model()

    # Student information
    student = {
        "name": "John Doe",
        "cgpa": 7.5,
        "year_of_study": 3,
        "domain": "Computer Science",
        "backlogs": 1,
        "skills": [
            "Python",
            "Machine Learning",
            "SQL"
        ],
        "skills_count": 3,
        "has_experience": 1
    }

    # Generate eligibility report
    checker.generate_report(student)

    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETED")
    print("=" * 60)
