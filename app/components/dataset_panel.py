"""
Dataset selection panel - Connected to existing DATA/datasets.py
"""
import streamlit as st
import sys
sys.path.append('/app/app/utils')
from app.utils.data_loader import dataset_service

def render():
    """Render the dataset selection panel component"""
    st.sidebar.header("📊 Dataset Selection")
    
    try:
        datasets = dataset_service.get_available_datasets()
        if not datasets:
            st.sidebar.error("No datasets found in DATA/custom_pairs/")
            return
        
        valid_datasets = {d['name']: d for d in datasets if not d.get('error', False)}
        
        if not valid_datasets:
            st.sidebar.error("No valid datasets found")
            return
        
        dataset_names = ["Select a dataset..."] + list(valid_datasets.keys())
        
        # Use a simpler selectbox
        selected_name = st.sidebar.selectbox(
            "Choose Dataset:",
            dataset_names,
            index=0,
            help="Select a dataset for analysis"
        )
        
        # If a valid dataset is selected, load it and update session state
        if selected_name != "Select a dataset...":
            selected_dataset = valid_datasets[selected_name]
            
            # Update session state only if the dataset has changed
            if st.session_state.get('selected_dataset', {}).get('id') != selected_dataset['id']:
                st.session_state.selected_dataset = selected_dataset
                with st.spinner("Loading dataset..."):
                    df = dataset_service.load_dataset(selected_dataset['id'])
                    st.session_state.current_dataframe = df
                    # Clear downstream results when dataset changes
                    st.session_state.pop('segmentation_result', None)
                    st.session_state.pop('causal_result', None)
                    st.session_state.pop('llm_result', None)
                st.sidebar.success("✅ Dataset loaded!")

        # Display info about the currently loaded dataset
        if 'current_dataframe' in st.session_state and st.session_state.current_dataframe is not None:
            df = st.session_state.current_dataframe
            selected_dataset = st.session_state.selected_dataset
            st.sidebar.subheader("Dataset Info")
            st.sidebar.write(f"**ID:** {selected_dataset['id']}")
            st.sidebar.write(f"**File:** {selected_dataset['file']}")
            st.sidebar.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} cols")
            st.sidebar.write(f"**Variables:** {', '.join(df.columns)}")
            
            with st.sidebar.expander("Preview Data"):
                st.dataframe(df.head(3))
        else:
            st.sidebar.info("Please select a dataset to begin.")

    except Exception as e:
        st.sidebar.error(f"❌ An error occurred: {e}")