"""
LLM-Powered Customer Persona Generation
Uses OpenAI GPT to generate natural language personas for customer segments
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional
import openai
from openai import OpenAI
import time
import warnings
warnings.filterwarnings('ignore')

class PersonaGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the persona generator
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use
        """
        self.model = model
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)
        
        self.personas = {}
        
    def load_cluster_data(self, cluster_stats_path: str, cluster_sizes_path: str = None):
        """
        Load cluster statistics and sizes
        
        Args:
            cluster_stats_path: Path to cluster statistics CSV
            cluster_sizes_path: Path to cluster sizes CSV (optional)
        """
        self.cluster_stats = pd.read_csv(cluster_stats_path, index_col=0)
        
        if cluster_sizes_path and os.path.exists(cluster_sizes_path):
            self.cluster_sizes = pd.read_csv(cluster_sizes_path, index_col=0)
        else:
            # Extract cluster sizes from cluster_stats if not provided separately
            self.cluster_sizes = None
            
        print(f"Loaded cluster data for {len(self.cluster_stats)} clusters")
        return self.cluster_stats
    
    def create_persona_prompt(self, cluster_id: int, cluster_stats: pd.Series, cluster_size: int, total_customers: int) -> str:
        """
        Create a detailed prompt for persona generation
        
        Args:
            cluster_id: Cluster identifier
            cluster_stats: Statistics for the cluster
            cluster_size: Number of customers in cluster
            total_customers: Total number of customers
            
        Returns:
            Formatted prompt string
        """
        cluster_percentage = (cluster_size / total_customers) * 100
        
        prompt = f"""You are an expert marketing strategist and customer insights analyst. I need you to create a detailed customer persona based on behavioral data from an e-commerce platform.

CLUSTER DATA:
- Cluster ID: {cluster_id}
- Size: {cluster_size:,} customers ({cluster_percentage:.1f}% of total customer base)

BEHAVIORAL CHARACTERISTICS:
- Recency (days since last purchase): {cluster_stats.get('recency_days_mean', 'N/A'):.0f} days (avg)
- Frequency (total orders): {cluster_stats.get('frequency_mean', 'N/A'):.1f} orders
- Monetary Value: ${cluster_stats.get('monetary_value_mean', 'N/A'):.2f} total spent
- Average Order Value: ${cluster_stats.get('avg_order_value_mean', 'N/A'):.2f}
- Total Items Purchased: {cluster_stats.get('total_items_mean', 'N/A'):.1f} items
- Average Items per Order: {cluster_stats.get('avg_items_per_order_mean', 'N/A'):.1f}
- Average Days Between Orders: {cluster_stats.get('avg_days_between_orders_mean', 'N/A'):.0f} days
- Total Freight Paid: ${cluster_stats.get('total_freight_mean', 'N/A'):.2f}

Please create a comprehensive customer persona that includes:

1. PERSONA NAME: A catchy, descriptive name (e.g., "The Savvy Mobile Shopper", "The Premium Loyalist")

2. DEMOGRAPHIC PROFILE: Age range, likely income level, lifestyle characteristics

3. BEHAVIORAL PATTERNS: Shopping frequency, spending patterns, product preferences

4. PSYCHOGRAPHIC INSIGHTS: Motivations, values, shopping preferences

5. MARKETING RECOMMENDATIONS: Specific strategies to target this segment

6. BUSINESS VALUE: Why this segment is important to the business

Format your response as a structured persona that a marketing team could immediately use for campaign planning. Be specific, actionable, and business-focused."""

        return prompt
    
    def generate_persona(self, cluster_id: int, cluster_stats: pd.Series, cluster_size: int, total_customers: int) -> Dict:
        """
        Generate a persona for a specific cluster
        
        Args:
            cluster_id: Cluster identifier
            cluster_stats: Statistics for the cluster
            cluster_size: Number of customers in cluster
            total_customers: Total number of customers
            
        Returns:
            Dictionary containing persona information
        """
        print(f"Generating persona for Cluster {cluster_id}...")
        
        # Create prompt
        prompt = self.create_persona_prompt(cluster_id, cluster_stats, cluster_size, total_customers)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert marketing strategist specializing in customer segmentation and persona development."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            persona_text = response.choices[0].message.content
            
            # Parse the response to extract structured information
            persona_data = self.parse_persona_response(persona_text, cluster_id, cluster_stats, cluster_size, total_customers)
            
            # Add a small delay to respect rate limits
            time.sleep(1)
            
            return persona_data
            
        except Exception as e:
            print(f"Error generating persona for Cluster {cluster_id}: {str(e)}")
            return {
                'cluster_id': cluster_id,
                'persona_name': f"Cluster {cluster_id}",
                'persona_text': f"Error generating persona: {str(e)}",
                'demographics': "Unable to generate",
                'behavioral_patterns': "Unable to generate",
                'psychographics': "Unable to generate",
                'marketing_recommendations': "Unable to generate",
                'business_value': "Unable to generate"
            }
    
    def parse_persona_response(self, persona_text: str, cluster_id: int, cluster_stats: pd.Series, cluster_size: int, total_customers: int) -> Dict:
        """
        Parse the LLM response to extract structured persona information
        
        Args:
            persona_text: Raw persona text from LLM
            cluster_id: Cluster identifier
            cluster_stats: Statistics for the cluster
            cluster_size: Number of customers in cluster
            total_customers: Total number of customers
            
        Returns:
            Structured persona dictionary
        """
        # Extract persona name (usually the first line or after "PERSONA NAME:")
        lines = persona_text.split('\n')
        persona_name = f"Cluster {cluster_id}"
        
        for line in lines:
            if 'PERSONA NAME:' in line.upper():
                persona_name = line.split(':', 1)[1].strip()
                break
            elif line.strip() and not line.startswith(' ') and len(line.strip()) < 50:
                # Likely the persona name if it's a short, non-indented line
                persona_name = line.strip()
                break
        
        # Try to extract sections (this is a simple approach - could be enhanced)
        sections = {
            'demographics': '',
            'behavioral_patterns': '',
            'psychographics': '',
            'marketing_recommendations': '',
            'business_value': ''
        }
        
        current_section = None
        for line in lines:
            line_lower = line.lower()
            if 'demographic' in line_lower:
                current_section = 'demographics'
            elif 'behavioral' in line_lower or 'behavior' in line_lower:
                current_section = 'behavioral_patterns'
            elif 'psychographic' in line_lower or 'motivation' in line_lower:
                current_section = 'psychographics'
            elif 'marketing' in line_lower or 'recommendation' in line_lower:
                current_section = 'marketing_recommendations'
            elif 'business' in line_lower or 'value' in line_lower:
                current_section = 'business_value'
            
            if current_section and line.strip() and not line.lower().startswith(('demographic', 'behavioral', 'psychographic', 'marketing', 'business')):
                sections[current_section] += line.strip() + ' '
        
        return {
            'cluster_id': cluster_id,
            'persona_name': persona_name,
            'persona_text': persona_text,
            'cluster_size': cluster_size,
            'cluster_percentage': (cluster_size / total_customers) * 100,
            'key_metrics': {
                'recency_days': cluster_stats.get('recency_days_mean', 0),
                'frequency': cluster_stats.get('frequency_mean', 0),
                'monetary_value': cluster_stats.get('monetary_value_mean', 0),
                'avg_order_value': cluster_stats.get('avg_order_value_mean', 0),
                'total_items': cluster_stats.get('total_items_mean', 0)
            },
            'demographics': sections['demographics'].strip(),
            'behavioral_patterns': sections['behavioral_patterns'].strip(),
            'psychographics': sections['psychographics'].strip(),
            'marketing_recommendations': sections['marketing_recommendations'].strip(),
            'business_value': sections['business_value'].strip()
        }
    
    def generate_all_personas(self, total_customers: int = None) -> Dict:
        """
        Generate personas for all clusters
        
        Args:
            total_customers: Total number of customers (if None, will estimate from cluster sizes)
            
        Returns:
            Dictionary of all generated personas
        """
        if total_customers is None:
            # Estimate total customers from cluster sizes
            total_customers = self.cluster_stats.shape[0] * 1000  # Rough estimate
        
        print(f"Generating personas for {len(self.cluster_stats)} clusters...")
        
        all_personas = {}
        
        for cluster_id in self.cluster_stats.index:
            cluster_stats = self.cluster_stats.loc[cluster_id]
            
            # Get cluster size
            if self.cluster_sizes is not None and cluster_id in self.cluster_sizes.index:
                cluster_size = self.cluster_sizes.loc[cluster_id, 'count']
            else:
                # Estimate cluster size (this is a rough approximation)
                cluster_size = int(total_customers / len(self.cluster_stats))
            
            # Generate persona
            persona = self.generate_persona(cluster_id, cluster_stats, cluster_size, total_customers)
            all_personas[cluster_id] = persona
            
            print(f"✓ Generated persona for Cluster {cluster_id}: {persona['persona_name']}")
        
        self.personas = all_personas
        return all_personas
    
    def save_personas(self, output_path: str = "data/customer_personas.json"):
        """
        Save generated personas to JSON file
        
        Args:
            output_path: Path to save the personas
        """
        if not self.personas:
            print("No personas to save. Run generate_all_personas() first.")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.personas, f, indent=2, ensure_ascii=False)
        
        print(f"Personas saved to {output_path}")
    
    def create_persona_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all personas
        
        Returns:
            DataFrame with persona summaries
        """
        if not self.personas:
            print("No personas available. Run generate_all_personas() first.")
            return pd.DataFrame()
        
        summary_data = []
        for cluster_id, persona in self.personas.items():
            summary_data.append({
                'cluster_id': cluster_id,
                'persona_name': persona['persona_name'],
                'cluster_size': persona['cluster_size'],
                'cluster_percentage': persona['cluster_percentage'],
                'recency_days': persona['key_metrics']['recency_days'],
                'frequency': persona['key_metrics']['frequency'],
                'monetary_value': persona['key_metrics']['monetary_value'],
                'avg_order_value': persona['key_metrics']['avg_order_value'],
                'total_items': persona['key_metrics']['total_items']
            })
        
        return pd.DataFrame(summary_data)
    
    def display_persona(self, cluster_id: int):
        """
        Display a formatted persona
        
        Args:
            cluster_id: Cluster identifier
        """
        if cluster_id not in self.personas:
            print(f"No persona found for Cluster {cluster_id}")
            return
        
        persona = self.personas[cluster_id]
        
        print("=" * 80)
        print(f"PERSONA: {persona['persona_name']}")
        print("=" * 80)
        print(f"Cluster ID: {cluster_id}")
        print(f"Size: {persona['cluster_size']:,} customers ({persona['cluster_percentage']:.1f}%)")
        print()
        
        print("KEY METRICS:")
        metrics = persona['key_metrics']
        print(f"  • Recency: {metrics['recency_days']:.0f} days since last purchase")
        print(f"  • Frequency: {metrics['frequency']:.1f} total orders")
        print(f"  • Monetary Value: ${metrics['monetary_value']:.2f} total spent")
        print(f"  • Avg Order Value: ${metrics['avg_order_value']:.2f}")
        print(f"  • Total Items: {metrics['total_items']:.1f} items")
        print()
        
        if persona['demographics']:
            print("DEMOGRAPHICS:")
            print(f"  {persona['demographics']}")
            print()
        
        if persona['behavioral_patterns']:
            print("BEHAVIORAL PATTERNS:")
            print(f"  {persona['behavioral_patterns']}")
            print()
        
        if persona['psychographics']:
            print("PSYCHOGRAPHICS:")
            print(f"  {persona['psychographics']}")
            print()
        
        if persona['marketing_recommendations']:
            print("MARKETING RECOMMENDATIONS:")
            print(f"  {persona['marketing_recommendations']}")
            print()
        
        if persona['business_value']:
            print("BUSINESS VALUE:")
            print(f"  {persona['business_value']}")
            print()

def main():
    """
    Main function to demonstrate persona generation
    Note: Requires OPENAI_API_KEY environment variable to be set
    """
    print("Customer Persona Generation with LLM")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY environment variable not set.")
        print("To use this feature, you need to:")
        print("1. Get an API key from https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable: export OPENAI_API_KEY='your-key-here'")
        print("3. Or pass it directly to PersonaGenerator(api_key='your-key-here')")
        return
    
    try:
        # Initialize persona generator
        generator = PersonaGenerator()
        
        # Load cluster data (you'll need to run clustering analysis first)
        cluster_stats_path = "data/cluster_analysis_kmeans.csv"
        
        if not os.path.exists(cluster_stats_path):
            print(f"❌ Cluster statistics file not found: {cluster_stats_path}")
            print("Please run the clustering analysis first to generate cluster data.")
            return
        
        generator.load_cluster_data(cluster_stats_path)
        
        # Generate personas for all clusters
        personas = generator.generate_all_personas(total_customers=96096)  # From our data exploration
        
        # Save personas
        generator.save_personas()
        
        # Create summary
        summary_df = generator.create_persona_summary()
        summary_df.to_csv("data/persona_summary.csv", index=False)
        
        print("\n" + "=" * 50)
        print("PERSONA GENERATION COMPLETE!")
        print("=" * 50)
        print(f"Generated {len(personas)} personas")
        print("Files saved:")
        print("  - data/customer_personas.json (detailed personas)")
        print("  - data/persona_summary.csv (summary table)")
        
        # Display first persona as example
        if personas:
            first_cluster = list(personas.keys())[0]
            generator.display_persona(first_cluster)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have:")
        print("1. Valid OpenAI API key")
        print("2. Cluster analysis results in data/cluster_analysis_*.csv")
        print("3. Internet connection for API calls")

if __name__ == "__main__":
    main()
