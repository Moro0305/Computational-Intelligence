import joblib
import pandas as pd
import numpy as np
import os
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
warnings.filterwarnings('ignore')


class CareerPredictor:
    """Handles career predictions using the trained model"""

    def __init__(self, model_path=None):
        """Load the trained model from file"""
        if model_path is None:
            # Search for model in common locations
            possible_paths = [
                'cs_students_career_predictor_model.pkl',
                '../Export ML/cs_students_career_predictor_model.pkl',
                '../cs_students_career_predictor_model.pkl',
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                raise FileNotFoundError("Model file 'cs_students_career_predictor_model.pkl' not found")

        self.model = joblib.load(model_path)
        print(f"Model loaded: {os.path.basename(model_path)}\n")

    def predict(self, gpa, projects, python_skill, sql_skill, java_skill, interested_domain=None):
        """
        Make a career prediction

        Args:
            gpa: Student GPA (0.0-4.0)
            projects: Project type (e.g., "Machine Learning")
            python_skill: Python level ("Strong", "Average", or "Weak")
            sql_skill: SQL level ("Strong", "Average", or "Weak")
            java_skill: Java level ("Strong", "Average", or "Weak")
            interested_domain: Area of interest (optional, will be inferred if None)
        """
        # Use placeholder if domain not specified
        if interested_domain is None or interested_domain.strip() == "":
            interested_domain = "Not Specified"

        # Create input dataframe
        student_data = pd.DataFrame({
            'GPA': [gpa],
            'Interested Domain': [interested_domain],
            'Projects': [projects],
            'Python': [python_skill],
            'SQL': [sql_skill],
            'Java': [java_skill]
        })

        # Get predictions and probabilities
        prediction = self.model.predict(student_data)[0]
        probabilities = self.model.predict_proba(student_data)[0]
        confidence = probabilities.max()

        # Get top 3 predictions
        classes = self.model.named_steps['classifier'].classes_
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3 = [(classes[i], probabilities[i]) for i in top_3_indices]

        return prediction, confidence, top_3


# Common options for dropdowns
INTERESTED_DOMAINS = [
    "Not Specified (will be inferred)",
    "Artificial Intelligence",
    "Web Development",
    "Data Science",
    "Cybersecurity",
    "Machine Learning",
    "Cloud Computing",
    "Mobile App Development",
    "Database Management",
    "Software Development",
    "Computer Graphics",
    "Network Security",
    "Software Engineering",
    "Natural Language Processing",
    "Computer Vision",
    "IoT (Internet of Things)",
    "Blockchain Technology",
    "Game Development",
    "Human-Computer Interaction",
    "Bioinformatics",
    "Quantum Computing",
    "Distributed Systems",
    "Digital Forensics",
    "Data Privacy",
    "Geographic Information Systems"
]

PROJECT_TYPES = [
    "Machine Learning",
    "Natural Language Processing",
    "E-commerce Website",
    "Network Security",
    "Mobile App",
    "Data Analytics",
    "Chatbot Development",
    "Full-Stack Web App",
    "SQL Query Optimization",
    "AWS Deployment",
    "GCP Deployment",
    "Cloud Migration",
    "Android App",
    "iOS App",
    "3D Rendering",
    "3D Animation",
    "3D Modeling",
    "Image Recognition",
    "Image Classification",
    "Game Development",
    "Social Media Platform",
    "Embedded Systems",
    "Deep Learning Models",
    "Data Warehouse Design",
    "Front-End Development",
    "Statistical Analysis",
    "Robotics",
    "Object Detection",
    "DevOps",
    "Genomic Data Analysis",
    "Smart Home Automation",
    "Usability Testing",
    "Medical Imaging Analysis",
    "Quantum Algorithm Development",
    "Virtual Reality Development",
    "Smart Contracts Developer",
    "Search Engine Optimization",
    "GIS Mapping",
    "Computer Forensic Analysis",
    "Protein Structure Prediction",
    "Neural Network Development",
    "Big Data Analytics",
    "Firewall Management",
    "Penetration Testing",
    "Security Auditing"
]

SKILL_LEVELS = ["Strong", "Average", "Weak"]


class PredictionGUI:
    """Simple GUI for career prediction input"""

    def __init__(self, predictor):
        """Initialize the GUI"""
        self.predictor = predictor
        self.result = None

        # Create main window
        self.root = tk.Tk()
        self.root.title("CS Students Career Prediction")
        self.root.geometry("600x550")
        self.root.resizable(False, False)

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="CS Students Career Prediction",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # GPA
        ttk.Label(main_frame, text="GPA (0.0-4.0):", font=('Arial', 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.gpa_var = tk.StringVar(value="3.5")
        gpa_spinbox = ttk.Spinbox(main_frame, from_=0.0, to=4.0, increment=0.1,
                                  textvariable=self.gpa_var, width=40)
        gpa_spinbox.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Interested Domain
        ttk.Label(main_frame, text="Interested Domain:", font=('Arial', 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.domain_var = tk.StringVar(value=INTERESTED_DOMAINS[0])
        domain_combo = ttk.Combobox(main_frame, textvariable=self.domain_var,
                                   values=INTERESTED_DOMAINS, state='readonly', width=38)
        domain_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)

        # Project Type
        ttk.Label(main_frame, text="Project Type:", font=('Arial', 10)).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.project_var = tk.StringVar(value=PROJECT_TYPES[0])
        project_combo = ttk.Combobox(main_frame, textvariable=self.project_var,
                                    values=PROJECT_TYPES, state='readonly', width=38)
        project_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)

        # Programming Skills Section
        ttk.Label(main_frame, text="Programming Skills",
                 font=('Arial', 11, 'bold')).grid(row=4, column=0, columnspan=2, pady=(15, 10))

        # Python
        ttk.Label(main_frame, text="Python:", font=('Arial', 10)).grid(row=5, column=0, sticky=tk.W, pady=5)
        self.python_var = tk.StringVar(value="Strong")
        python_combo = ttk.Combobox(main_frame, textvariable=self.python_var,
                                   values=SKILL_LEVELS, state='readonly', width=38)
        python_combo.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5)

        # SQL
        ttk.Label(main_frame, text="SQL:", font=('Arial', 10)).grid(row=6, column=0, sticky=tk.W, pady=5)
        self.sql_var = tk.StringVar(value="Average")
        sql_combo = ttk.Combobox(main_frame, textvariable=self.sql_var,
                                values=SKILL_LEVELS, state='readonly', width=38)
        sql_combo.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5)

        # Java
        ttk.Label(main_frame, text="Java:", font=('Arial', 10)).grid(row=7, column=0, sticky=tk.W, pady=5)
        self.java_var = tk.StringVar(value="Weak")
        java_combo = ttk.Combobox(main_frame, textvariable=self.java_var,
                                 values=SKILL_LEVELS, state='readonly', width=38)
        java_combo.grid(row=7, column=1, sticky=(tk.W, tk.E), pady=5)

        # Predict button
        predict_btn = ttk.Button(main_frame, text="Predict Career", command=self.predict)
        predict_btn.grid(row=8, column=0, columnspan=2, pady=(20, 10))

        # Status label
        self.status_var = tk.StringVar(value="Ready to predict")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                font=('Arial', 9, 'italic'), foreground='gray')
        status_label.grid(row=9, column=0, columnspan=2)

        # Configure column weights for responsiveness
        main_frame.columnconfigure(1, weight=1)

    def predict(self):
        """Handle prediction button click"""
        try:
            # Validate GPA
            gpa = float(self.gpa_var.get())
            if gpa < 0.0 or gpa > 4.0:
                messagebox.showerror("Invalid Input", "GPA must be between 0.0 and 4.0")
                return

            # Get interested domain (handle "Not Specified")
            domain = self.domain_var.get()
            if domain.startswith("Not Specified"):
                domain = None

            # Update status
            self.status_var.set("Processing prediction...")
            self.root.update()

            # Make prediction
            prediction, confidence, top_3 = self.predictor.predict(
                gpa=gpa,
                projects=self.project_var.get(),
                python_skill=self.python_var.get(),
                sql_skill=self.sql_var.get(),
                java_skill=self.java_var.get(),
                interested_domain=domain
            )

            # Store results and close window
            self.result = {
                'prediction': prediction,
                'confidence': confidence,
                'top_3': top_3,
                'student_info': {
                    'gpa': gpa,
                    'domain': domain,
                    'projects': self.project_var.get(),
                    'python': self.python_var.get(),
                    'sql': self.sql_var.get(),
                    'java': self.java_var.get()
                }
            }

            self.status_var.set("Prediction complete!")
            self.root.after(500, self.root.destroy)

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter a valid GPA number\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def run(self):
        """Run the GUI and return results"""
        self.root.mainloop()
        return self.result


def get_input(prompt, input_type=str, valid_options=None, allow_empty=False):
    """Get and validate user input"""
    while True:
        try:
            value = input(prompt).strip()

            # Allow empty input if specified
            if allow_empty and value == "":
                return None

            # Convert to appropriate type
            if input_type == float:
                return float(value)
            elif input_type == int:
                return int(value)

            # Validate against options if provided
            if valid_options:
                if value.lower() in [opt.lower() for opt in valid_options]:
                    # Return with proper casing
                    for opt in valid_options:
                        if opt.lower() == value.lower():
                            return opt
                print(f"  Invalid input. Choose from: {', '.join(valid_options)}")
                continue

            return value

        except ValueError:
            print(f"  Invalid input. Please enter a valid {input_type.__name__}")
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            exit(0)


def print_results(prediction, confidence, top_3, student_info):
    """Display prediction results"""
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)

    # Show student info
    print("\nStudent Profile:")
    print(f"  GPA: {student_info['gpa']:.2f}")
    print(f"  Interested Domain: {student_info['domain'] or 'Not Specified (inferred)'}")
    print(f"  Projects: {student_info['projects']}")
    print(f"  Skills: Python={student_info['python']}, SQL={student_info['sql']}, Java={student_info['java']}")

    # Show main prediction
    print(f"\nPredicted Career: {prediction}")
    print(f"Confidence: {confidence*100:.1f}%")

    # Show top 3 alternatives
    print("\nTop 3 Recommendations:")
    for i, (career, prob) in enumerate(top_3, 1):
        bar = "#" * int(prob * 50)
        print(f"  {i}. {career:<40} {bar} {prob*100:5.1f}%")

    print("="*70 + "\n")


def main():
    """Main application"""
    print("\n" + "="*70)
    print("CS STUDENTS CAREER PREDICTION")
    print("="*70 + "\n")

    try:
        # Load model
        predictor = CareerPredictor()

        # Launch GUI
        print("Opening GUI for input selection...\n")
        gui = PredictionGUI(predictor)
        result = gui.run()

        # Check if user provided input
        if result is None:
            print("Prediction cancelled by user.\n")
            return

        # Display results in terminal
        print_results(
            result['prediction'],
            result['confidence'],
            result['top_3'],
            result['student_info']
        )

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the model file exists or run Training_Export.ipynb first.\n")
        messagebox.showerror("Model Not Found", str(e))
    except Exception as e:
        print(f"\nError: {e}\n")
        messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    main()
