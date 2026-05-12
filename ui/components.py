"""
UI Components Module
====================
Streamlit UI components for the application.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def render_sidebar() -> dict:
    """Render sidebar configuration and return settings."""

    st.sidebar.title("⚙️ Configuration")

    # Model selection
    st.sidebar.subheader("🤖 Models to Train")
    models = []

    model_options = {
        'Logistic Regression': True,
        'Decision Tree': True,
        'Random Forest': True,
        'Gradient Boosting': True,
        'SVM': True,
        'KNN': True,
        'Naive Bayes': True
    }

    for model, default in model_options.items():
        if st.sidebar.checkbox(model, value=default, key=f"model_{model}"):
            models.append(model)

    if not models:
        st.sidebar.warning("Please select at least one model")

    # PCA settings
    st.sidebar.subheader("🎯 PCA Settings")
    variance_threshold = st.sidebar.slider(
        "Variance Threshold",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        help="Minimum cumulative variance to retain"
    )

    # Comparison settings
    st.sidebar.subheader("📊 Comparison Settings")
    compare_before_pca = st.sidebar.checkbox(
        "Compare with original data",
        value=True,
        help="Train models on original data for comparison"
    )

    st.sidebar.divider()

    # About section
    st.sidebar.info("""
    **PCA Performance Optimizer**

    Project-Based Learning — Machine Learning (10300401)

    Upload a dataset, apply PCA, and compare model performance.
    """)

    return {
        'models': models,
        'variance_threshold': variance_threshold,
        'compare_before_pca': compare_before_pca
    }


def render_results(comparison: dict, pca_results: dict, model_results: dict = None):
    """Render comparison results and visualizations."""

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Accuracy Comparison",
        "⏱️ Training & Inference Time",
        "💾 Memory Usage",
        "🔢 Confusion Matrices",
        "📊 Variance Analysis"
    ])

    with tab1:
        render_accuracy_chart(comparison)

    with tab2:
        render_time_comparison(comparison)

    with tab3:
        render_memory_comparison(comparison)

    with tab4:
        render_confusion_matrices(comparison, model_results or {})

    with tab5:
        render_variance_analysis(pca_results)


def render_accuracy_chart(comparison: dict):
    """Render accuracy comparison bar chart."""

    models = comparison['models']
    accuracy_before = [comparison['metrics']['accuracy']['before_pca'].get(m, 0) for m in models]
    accuracy_after = [comparison['metrics']['accuracy']['after_pca'].get(m, 0) for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before PCA',
        x=models,
        y=accuracy_before,
        marker_color='lightcoral'
    ))
    fig.add_trace(go.Bar(
        name='After PCA',
        x=models,
        y=accuracy_after,
        marker_color='mediumseagreen'
    ))

    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        barmode='group',
        yaxis_tickformat='.2%'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show accuracy table
    st.write("### Accuracy Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Before PCA", f"{max(accuracy_before):.2%}")
    with col2:
        st.metric("Best After PCA", f"{max(accuracy_after):.2%}")
    with col3:
        diff = max(accuracy_after) - max(accuracy_before)
        st.metric("Difference", f"{diff:+.2%}")


def render_time_comparison(comparison: dict):
    """Render training and inference time comparison."""

    models = comparison['models']

    # Training time
    train_before = [comparison['metrics']['training_time']['before_pca'].get(m, 0) for m in models]
    train_after = [comparison['metrics']['training_time']['after_pca'].get(m, 0) for m in models]

    fig_train = go.Figure()
    fig_train.add_trace(go.Bar(
        name='Before PCA',
        x=models,
        y=train_before,
        marker_color='lightblue'
    ))
    fig_train.add_trace(go.Bar(
        name='After PCA',
        x=models,
        y=train_after,
        marker_color='darkblue'
    ))

    fig_train.update_layout(
        title="Training Time Comparison",
        xaxis_title="Model",
        yaxis_title="Time (seconds)",
        barmode='group'
    )

    st.plotly_chart(fig_train, use_container_width=True)

    # Inference time
    inf_before = [comparison['metrics']['inference_time']['before_pca'].get(m, 0) for m in models]
    inf_after = [comparison['metrics']['inference_time']['after_pca'].get(m, 0) for m in models]

    fig_inf = go.Figure()
    fig_inf.add_trace(go.Bar(
        name='Before PCA',
        x=models,
        y=inf_before,
        marker_color='lightsalmon'
    ))
    fig_inf.add_trace(go.Bar(
        name='After PCA',
        x=models,
        y=inf_after,
        marker_color='darkred'
    ))

    fig_inf.update_layout(
        title="Inference Time Comparison",
        xaxis_title="Model",
        yaxis_title="Time (seconds)",
        barmode='group'
    )

    st.plotly_chart(fig_inf, use_container_width=True)


def render_memory_comparison(comparison: dict):
    """Render memory usage comparison."""

    models = comparison['models']
    mem_before = [comparison['metrics']['memory_used']['before_pca'].get(m, 0) for m in models]
    mem_after = [comparison['metrics']['memory_used']['after_pca'].get(m, 0) for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before PCA',
        x=models,
        y=mem_before,
        marker_color='lightgreen'
    ))
    fig.add_trace(go.Bar(
        name='After PCA',
        x=models,
        y=mem_after,
        marker_color='darkgreen'
    ))

    fig.update_layout(
        title="Memory Usage Comparison",
        xaxis_title="Model",
        yaxis_title="Memory (MB)",
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrices(comparison: dict, model_results: dict):
    """Render confusion matrices for each model."""

    if 'after_pca' not in model_results:
        st.info("Train models first to see confusion matrices")
        return

    for model_name in comparison['models']:
        if model_name in model_results['after_pca']:
            result = model_results['after_pca'][model_name]
            cm = result.get('confusion_matrix')
            y_test = result.get('y_test')

            if cm is not None and y_test is not None:
                st.subheader(f"📊 {model_name}")

                # Create confusion matrix plot using matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))

                # Get class labels
                n_classes = len(cm)
                class_labels = [f'C{i}' for i in range(n_classes)]

                # Plot using matplotlib
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.figure.colorbar(im, ax=ax)

                ax.set(xticks=np.arange(n_classes),
                       yticks=np.arange(n_classes),
                       xticklabels=class_labels,
                       yticklabels=class_labels,
                       title=f'{model_name} Confusion Matrix',
                       ylabel='True Label',
                       xlabel='Predicted Label')

                # Rotate x-axis labels for better readability
                ax.set_xticklabels(class_labels, rotation=45, ha='right')
                ax.set_yticklabels(class_labels, rotation=0)

                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(n_classes):
                    for j in range(n_classes):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")

                fig.tight_layout()
                st.pyplot(fig)

                # Show accuracy and per-class metrics
                acc = result.get('accuracy', 0)
                precision = result.get('precision', 0)
                recall = result.get('recall', 0)
                f1 = result.get('f1_score', 0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{acc:.2%}")
                with col2:
                    st.metric("Precision", f"{precision:.2%}")
                with col3:
                    st.metric("Recall", f"{recall:.2%}")
                with col4:
                    st.metric("F1-Score", f"{f1:.2%}")

                # Show class distribution info
                st.caption(f"Class distribution: {n_classes} classes | Total predictions: {cm.sum()}")

                # Show classification report
                report = result.get('classification_report', '')
                if report:
                    with st.expander("📋 Detailed Classification Report"):
                        st.text(report)

                st.divider()


def render_variance_analysis(pca_results: dict):
    """Render PCA variance analysis charts."""

    if not pca_results:
        st.warning("No PCA results to display. Apply PCA first.")
        return

    variance = pca_results.get('explained_variance', [])
    cumulative = pca_results.get('cumulative_variance', [])

    if len(variance) == 0:
        st.warning("Variance data not available")
        return

    components = list(range(1, len(variance) + 1))

    # Individual variance plot
    fig1 = px.bar(
        x=components,
        y=variance,
        labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
        title='Variance Explained by Each Component'
    )
    fig1.update_traces(marker_color='royalblue')
    st.plotly_chart(fig1, use_container_width=True)

    # Cumulative variance plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=components,
        y=cumulative,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='green', width=2)
    ))

    # Add threshold line
    fig2.add_hline(
        y=0.95,
        line_dash="dash",
        annotation_text="95% threshold",
        line_color="red"
    )

    fig2.update_layout(
        title='Cumulative Variance Explained',
        xaxis_title='Number of Components',
        yaxis_title='Cumulative Explained Variance',
        yaxis_tickformat='.1%'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Summary metrics
    st.write("### PCA Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_var = cumulative[-1] if len(cumulative) > 0 else 0
    with col1:
        st.metric("Components Used", len(variance))
    with col2:
        st.metric("Total Variance", f"{total_var:.2%}")
    with col3:
        # Find components for 95%
        n_95 = np.argmax(np.array(cumulative) >= 0.95) + 1 if len(cumulative) > 0 else 'N/A'
        st.metric("For 95% Variance", n_95)
    with col4:
        n_99 = np.argmax(np.array(cumulative) >= 0.99) + 1 if len(cumulative) > 0 else 'N/A'
        st.metric("For 99% Variance", n_99)


def render_confusion_matrix_plot(cm, model_name: str):
    """Render a single confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {model_name}')
    st.pyplot(fig)