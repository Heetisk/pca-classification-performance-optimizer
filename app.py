"""
PCA Performance Optimizer
========================
A web application for intelligent feature reduction and model performance comparison.
Project-Based Learning — Machine Learning (10300401)
"""

import streamlit as st
from core.preprocessor import DataPreprocessor
from core.pca_analyzer import PCAAnalyzer
from core.model_trainer import ModelTrainer
from core.performance_comparator import PerformanceComparator
from core.config import UI_CONFIG, PCA_CONFIG, CLASSIFICATION_CONFIG, PREPROCESSING_CONFIG
from ui.components import render_sidebar, render_results

def init_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'X_train_pca' not in st.session_state:
        st.session_state.X_train_pca = None
    if 'X_test_pca' not in st.session_state:
        st.session_state.X_test_pca = None
    if 'pca_results' not in st.session_state:
        st.session_state.pca_results = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}

def main():
    st.set_page_config(
        page_title="PCA Performance Optimizer",
        page_icon="📊",
        layout="wide"
    )

    init_session_state()

    st.title("📊 PCA + Classification Performance Optimizer")
    st.markdown("""
    ### Intelligent Feature Reduction and Model Performance Comparison using PCA

    This tool helps you:
    - Upload and preprocess datasets
    - Apply PCA for dimensional reduction
    - Compare ML model performance before and after PCA
    - Visualize results and get optimal recommendations
    """)

    # Sidebar configuration
    config = render_sidebar()

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with features and target column"
        )

        if uploaded_file:
            with st.spinner("Loading data..."):
                preprocessor = DataPreprocessor()
                df = preprocessor.load_data(uploaded_file)

                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(UI_CONFIG['data_preview_rows']))

                # Data preprocessing
                st.header("⚙️ Preprocessing")

                # Show missing values info
                missing_info = df.isnull().sum()
                missing_cols = missing_info[missing_info > 0]
                if len(missing_cols) > 0:
                    st.warning(f"⚠️ Found {len(missing_cols)} columns with missing values. Auto-filling with mean.")
                    with st.expander("📋 Missing Values Details"):
                        st.write(missing_cols)

                # Show recommended targets
                preprocessor_temp = DataPreprocessor()
                preprocessor_temp.df = df
                recommendations = preprocessor_temp.get_recommended_targets()

                st.write("**📌 Target Column**")
                target_col = st.selectbox(
                    "Select Target Column",
                    options=df.columns.tolist(),
                    help="Column to predict. App will auto-clean text and bin too-many-classes columns."
                )

                # Show info about selected target
                target_nunique = df[target_col].nunique()
                target_dtype = df[target_col].dtype
                st.caption(f"Selected: {target_col} | {target_nunique} unique values | Type: {target_dtype}")

                if target_nunique > CLASSIFICATION_CONFIG['max_classes_for_classification']:
                    st.info(f"🔧 Auto-binning: {target_col} has {target_nunique} values → will be binned into ~{PREPROCESSING_CONFIG['n_bins']} categories")
                elif df[target_col].dtype == 'object':
                    st.info(f"🔧 Auto-encoding: {target_col} contains text → will be encoded to numbers")

                test_size = st.slider(
                    "Test Set Size",
                    UI_CONFIG['test_size_min'],
                    UI_CONFIG['test_size_max'],
                    UI_CONFIG['test_size_default'],
                    UI_CONFIG['test_size_step']
                )
                random_state = st.number_input(
                    "Random State",
                    UI_CONFIG['random_state_min'],
                    UI_CONFIG['random_state_max'],
                    UI_CONFIG['random_state_default']
                )

                if st.button("🔄 Preprocess & Split Data", type="primary"):
                    try:
                        with st.spinner("Preprocessing data..."):
                            preprocessor.set_target(target_col)
                            X_train, X_test, y_train, y_test = preprocessor.split_data(
                                test_size=test_size,
                                random_state=random_state
                            )

                            st.session_state.data = df
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test

                            # Show cleaning info
                            cleaning_msg = f"✅ Data preprocessed! ({len(X_train)} train, {len(X_test)} test)"
                            if preprocessor.cleaning_info.get('actions'):
                                cleaning_msg += "\n**Auto-cleaning applied:**"
                                for action in preprocessor.cleaning_info['actions']:
                                    cleaning_msg += f"\n- {action}"
                            st.success(cleaning_msg)
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")

                # PCA Configuration
                if st.session_state.X_train is not None:
                    st.header("🎯 PCA Configuration")

                    n_features = st.session_state.X_train.shape[1]
                    max_components = min(n_features, PCA_CONFIG['max_components_limit'])

                    if max_components <= 1:
                        st.info("Dataset has only 1 feature. PCA cannot reduce dimensionality further.")
                    else:
                        default_components = min(10, max_components) if max_components > 1 else 1
                        n_components = st.slider(
                            "Number of PCA Components",
                            1, max_components, default_components,
                            help="Number of principal components to keep"
                        )

                    standardize = st.checkbox("Standardize Features", value=True, help="Standardize before PCA")

                    if st.button("🔍 Apply PCA"):
                        with st.spinner("Applying PCA..."):
                            pca_analyzer = PCAAnalyzer()

                            if standardize:
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(st.session_state.X_train)
                                X_test_scaled = scaler.transform(st.session_state.X_test)
                            else:
                                X_train_scaled = st.session_state.X_train
                                X_test_scaled = st.session_state.X_test

                            # Apply PCA
                            pca_results = pca_analyzer.fit_transform(
                                X_train_scaled, X_test_scaled, n_components
                            )

                            # Store pca transformed data
                            st.session_state.pca_results = pca_results
                            st.session_state.X_train_pca = pca_results['X_train_pca']
                            st.session_state.X_test_pca = pca_results['X_test_pca']

                            st.success("✅ PCA applied successfully!")

                    # Model Training
                    if st.session_state.X_train_pca is not None:
                        st.header("🤖 Model Training")

                        models = config['models']
                        compare_before_pca = config['compare_before_pca']

                        if st.button("🚀 Train All Models", type="primary"):
                            # Validate target is suitable for classification
                            y_unique = st.session_state.y_train.nunique()
                            if y_unique > CLASSIFICATION_CONFIG['max_classes_for_classification']:
                                st.error(f"❌ Target column has {y_unique} unique values (looks like regression). Please use a classification target with ≤{CLASSIFICATION_CONFIG['max_classes_for_classification']} classes.")
                            elif y_unique < CLASSIFICATION_CONFIG['min_classes']:
                                st.error("❌ Target column needs at least 2 classes.")
                            else:
                                with st.spinner("Training models... (this may take a while)"):
                                    trainer = ModelTrainer()

                                    # Train on original data
                                    if compare_before_pca:
                                        results_before = {}
                                        for name in models:
                                            result = trainer.train_model(
                                                name, st.session_state.X_train, st.session_state.X_test,
                                                st.session_state.y_train, st.session_state.y_test
                                            )
                                            results_before[name] = result

                                        st.session_state.model_results['before_pca'] = results_before

                                    # Train on PCA data
                                    results_after = {}
                                    for name in models:
                                        result = trainer.train_model(
                                            name, st.session_state.X_train_pca, st.session_state.X_test_pca,
                                            st.session_state.y_train, st.session_state.y_test
                                        )
                                        results_after[name] = result

                                    st.session_state.model_results['after_pca'] = results_after

                                    st.success("✅ All models trained!")

    with col2:
        # Results Display
        if st.session_state.model_results:
            st.header("📊 Performance Comparison")

            comparator = PerformanceComparator()
            comparison = comparator.compare(st.session_state.model_results)

            # Render comparison charts
            render_results(comparison, st.session_state.pca_results, st.session_state.model_results)

            # Optimal recommendation
            if 'after_pca' in st.session_state.model_results:
                st.header("💡 Recommendations")
                optimal = comparator.get_optimal_recommendation(
                    comparison,
                    st.session_state.pca_results.get('explained_variance', [])
                )
                st.info(f"""
                **Optimal Configuration:**
                - Best Model: {optimal['best_model']}
                - Recommended PCA Components: {optimal['recommended_components']}
                - Expected Accuracy: {optimal['expected_accuracy']:.2%}
                - Variance Explained: {optimal['variance_explained']:.2%}
                """)

if __name__ == "__main__":
    main()