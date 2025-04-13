# Conceptual System Architecture (University Project Phase)

This document describes the conceptual flow of the predictive maintenance system as implemented for the university project, focusing on local execution with CSV data.

```mermaid
graph TD
    subgraph Input
        A[Raw Data (predictive_maintainance.csv)]
    end

    subgraph Processing Pipeline (src/pipelines/training_pipeline.py)
        B[1. Load Raw Data] --> C{2. Clean Data};
        C --> D[3. Engineer Features];
        D --> E{4. Split Data (Train/Test)};
        E -- Train Data --> F[5a. Fit Preprocessor];
        F --> G[5b. Transform Train Data];
        E -- Test Data --> H[5c. Transform Test Data];
        G --> I[6a. Train Model];
        I --> J[6b. Evaluate Model];
        H --> J;  # Test Data used for Evaluation
    end

    subgraph Artifacts (models/)
        K[Saved Preprocessor (preprocessor.joblib)];
        L[Saved Model (final_failure_classifier.joblib)];
    end

    subgraph Prediction Pipeline (src/pipelines/prediction_pipeline.py)
        M[New Raw Data (Manual Input / CSV)] --> N[1. Load/Format Data];
        N --> O{2. Clean Data};
        O --> P[3. Engineer Features];
        P --> Q[4. Prepare Features];
        Q --> R[5. Transform Data];
        R --> S[6. Make Prediction];
    end

    subgraph User Interface (app/dashboard.py - Streamlit)
        T[Streamlit UI];
        T -- User Input --> M;
        S -- Prediction & Probability --> T;
        T -- Display --> U[User (Maintenance Personnel / Supervisor)];
    end

    subgraph Dependencies
        F --> K; # Preprocessor is saved
        I --> L; # Model is saved
        K --> R; # Prediction uses saved Preprocessor
        L --> S; # Prediction uses saved Model
    end

    Input --> Processing_Pipeline;
    Processing_Pipeline --> Artifacts;
    Artifacts --> Prediction_Pipeline;
    Prediction_Pipeline --> User_Interface;

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Artifacts fill:#ccf,stroke:#333,stroke-width:2px
    style User_Interface fill:#9cf,stroke:#333,stroke-width:2px