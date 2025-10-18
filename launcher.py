#!/usr/bin/env python3
"""
Customer Segmentation Project Launcher
Provides easy access to all project components
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print project banner"""
    print("=" * 80)
    print("🎯 CUSTOMER SEGMENTATION & PERSONA GENERATION PROJECT")
    print("=" * 80)
    print("FAANG-Ready Unsupervised ML with LLM Integration")
    print("=" * 80)

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "data/olist_customers_dataset.csv",
        "data/olist_orders_dataset.csv", 
        "data/olist_order_items_dataset.csv",
        "data/olist_order_payments_dataset.csv",
        "data/olist_products_dataset.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all Olist dataset files are in the data/ directory")
        return False
    
    print("✅ All required data files found")
    return True

def run_data_exploration():
    """Run data exploration and feature engineering"""
    print("\n🔍 Running Data Exploration & Feature Engineering...")
    try:
        subprocess.run([sys.executable, "src/data_exploration.py"], check=True)
        print("✅ Data exploration completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data exploration failed: {e}")
        return False

def run_clustering_analysis():
    """Run clustering analysis"""
    print("\n🎯 Running Clustering Analysis...")
    try:
        subprocess.run([sys.executable, "src/clustering_analysis.py"], check=True)
        print("✅ Clustering analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Clustering analysis failed: {e}")
        return False

def run_persona_generation():
    """Run persona generation"""
    print("\n🤖 Running Persona Generation...")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   Persona generation requires an OpenAI API key")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        subprocess.run([sys.executable, "src/persona_generation.py"], check=True)
        print("✅ Persona generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Persona generation failed: {e}")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("\n📊 Launching Interactive Dashboard...")
    try:
        subprocess.run(["streamlit", "run", "src/dashboard.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard launch failed: {e}")
        print("   Make sure Streamlit is installed: pip install streamlit")

def show_menu():
    """Show interactive menu"""
    while True:
        print("\n" + "=" * 50)
        print("📋 PROJECT MENU")
        print("=" * 50)
        print("1. 🔍 Run Data Exploration")
        print("2. 🎯 Run Clustering Analysis") 
        print("3. 🤖 Generate Customer Personas")
        print("4. 📊 Launch Dashboard")
        print("5. 🚀 Run Complete Pipeline")
        print("6. 📁 Show Project Structure")
        print("7. ❓ Help & Documentation")
        print("8. 🚪 Exit")
        print("=" * 50)
        
        choice = input("Select an option (1-8): ").strip()
        
        if choice == "1":
            if check_requirements():
                run_data_exploration()
        
        elif choice == "2":
            if Path("data/customer_features.csv").exists():
                run_clustering_analysis()
            else:
                print("❌ Please run data exploration first")
        
        elif choice == "3":
            if Path("data/cluster_analysis_kmeans.csv").exists():
                run_persona_generation()
            else:
                print("❌ Please run clustering analysis first")
        
        elif choice == "4":
            launch_dashboard()
        
        elif choice == "5":
            print("\n🚀 Running Complete Pipeline...")
            if check_requirements():
                if run_data_exploration():
                    if run_clustering_analysis():
                        run_persona_generation()
                        print("\n✅ Complete pipeline finished!")
                        print("   You can now launch the dashboard or view results")
        
        elif choice == "6":
            show_project_structure()
        
        elif choice == "7":
            show_help()
        
        elif choice == "8":
            print("\n👋 Thanks for using the Customer Segmentation Project!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-8.")

def show_project_structure():
    """Show project structure"""
    print("\n📁 PROJECT STRUCTURE")
    print("=" * 50)
    
    structure = """
customer_segmentation_project/
├── 📊 data/                          # Dataset files
│   ├── olist_*_dataset.csv          # Raw Olist data
│   ├── customer_features.csv         # Engineered features
│   ├── cluster_analysis_*.csv       # Clustering results
│   └── customer_personas.json        # AI-generated personas
├── 📁 src/                           # Source code
│   ├── data_exploration.py          # Data analysis & feature engineering
│   ├── clustering_analysis.py       # ML clustering algorithms
│   ├── persona_generation.py        # LLM persona creation
│   └── dashboard.py                 # Streamlit dashboard
├── 📁 plots/                         # Generated visualizations
├── 📁 notebooks/                     # Jupyter notebooks (optional)
├── 📁 docs/                          # Documentation
├── 📁 tests/                         # Unit tests
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── launcher.py                      # This launcher script
    """
    
    print(structure)

def show_help():
    """Show help and documentation"""
    print("\n❓ HELP & DOCUMENTATION")
    print("=" * 50)
    print("""
🎯 PROJECT OVERVIEW:
This project demonstrates customer segmentation using unsupervised ML
combined with LLM-powered persona generation for business insights.

📋 WORKFLOW:
1. Data Exploration → Feature Engineering → Customer Features
2. Clustering Analysis → K-means/DBSCAN → Customer Segments  
3. Persona Generation → OpenAI GPT → Marketing Personas
4. Dashboard → Streamlit → Interactive Visualizations

🔧 REQUIREMENTS:
- Python 3.8+
- OpenAI API key (for persona generation)
- All dependencies in requirements.txt

📊 OUTPUTS:
- Customer segments with behavioral insights
- AI-generated marketing personas
- Interactive dashboard for exploration
- Business recommendations for each segment

🚀 QUICK START:
1. Install dependencies: pip install -r requirements.txt
2. Set API key: export OPENAI_API_KEY='your-key'
3. Run complete pipeline: python launcher.py (option 5)
4. Launch dashboard: streamlit run src/dashboard.py

📚 DOCUMENTATION:
- README.md: Comprehensive project documentation
- Code comments: Detailed explanations in each module
- Dashboard: Interactive help and tooltips

🆘 TROUBLESHOOTING:
- Missing data files: Ensure Olist dataset is in data/ directory
- API errors: Check OpenAI API key and internet connection
- Import errors: Run pip install -r requirements.txt
- Dashboard issues: Check Streamlit installation
    """)

def main():
    """Main function"""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        print("   The src/ directory should be visible")
        return
    
    # Show menu
    show_menu()

if __name__ == "__main__":
    main()
