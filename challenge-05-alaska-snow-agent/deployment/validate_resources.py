"""
Resource validation script for Alaska Snow Agent
Checks if all required GCP resources exist before starting the app
"""

import os
import sys
from google.cloud import bigquery

def validate_environment():
    """Validate required environment variables and GCP resources."""

    print("🔍 Validating environment and GCP resources...")
    print("=" * 70)

    # 1. Check PROJECT_ID
    project_id = os.environ.get("PROJECT_ID")

    if not project_id:
        print("❌ ERROR: PROJECT_ID environment variable is required")
        print("   Set PROJECT_ID environment variable")
        print()
        print("   For Cloud Run deployment, use:")
        print("   --set-env-vars PROJECT_ID=your-project-id")
        return False

    print(f"✅ Project ID: {project_id}")

    # 2. Check REGION
    region = os.environ.get("REGION", "us-central1")
    print(f"✅ Region: {region}")

    # 3. Check DATASET_ID
    dataset_id = os.environ.get("DATASET_ID", "alaska_snow_capstone")
    print(f"✅ Dataset ID: {dataset_id}")

    print()
    print("🔍 Checking BigQuery resources...")

    try:
        client = bigquery.Client(project=project_id, location=region)

        # 4. Check dataset exists
        try:
            dataset = client.get_dataset(f"{project_id}.{dataset_id}")
            print(f"✅ Dataset '{dataset_id}' exists")
        except Exception as e:
            print(f"❌ ERROR: Dataset '{dataset_id}' not found")
            print(f"   {str(e)}")
            print()
            print("   To create the dataset, run the Challenge 5 notebook:")
            print(f"   - Cell 2: Create dataset")
            return False

        # 5. Check snow_vectors table exists
        try:
            table = client.get_table(f"{project_id}.{dataset_id}.snow_vectors")
            row_count = table.num_rows
            print(f"✅ Table 'snow_vectors' exists ({row_count} rows)")

            if row_count == 0:
                print("   ⚠️  WARNING: Table is empty - run notebook Cell 3 to generate embeddings")
        except Exception as e:
            print(f"❌ ERROR: Table 'snow_vectors' not found")
            print(f"   {str(e)}")
            print()
            print("   To create the table, run the Challenge 5 notebook:")
            print(f"   - Cell 3: Build vector search index")
            return False

        # 6. Check embedding_model exists
        try:
            model = client.get_model(f"{project_id}.{dataset_id}.embedding_model")
            print(f"✅ Model 'embedding_model' exists")
        except Exception as e:
            print(f"❌ ERROR: Model 'embedding_model' not found")
            print(f"   {str(e)}")
            print()
            print("   To create the model, run the Challenge 5 notebook:")
            print(f"   - Cell 3: Build vector search index")
            return False

    except Exception as e:
        print(f"❌ ERROR: Cannot connect to BigQuery")
        print(f"   {str(e)}")
        print()
        print("   Possible causes:")
        print("   - Missing IAM permissions (need bigquery.datasets.get)")
        print("   - Invalid project ID")
        print("   - BigQuery API not enabled")
        return False

    print()
    print("=" * 70)
    print("✅ All required resources validated successfully!")
    print()
    return True

if __name__ == "__main__":
    if not validate_environment():
        print()
        print("💡 SOLUTION:")
        print("   1. Open the Challenge 5 notebook in Google Colab")
        print("   2. Run all cells to create required BigQuery resources")
        print("   3. Ensure you're using the same PROJECT_ID")
        print()
        sys.exit(1)

    sys.exit(0)
