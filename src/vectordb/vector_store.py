import os
from typing import List, Dict
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm.auto import tqdm

class VectorDB:
    def __init__(self):
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise ValueError("❌ Environment variables SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found!")
        
        self.client: Client = create_client(url, key)

    def upload_chunks(self, rows: List[Dict], table_name: str = "chunks", batch_size: int = 100):
        """
        Uploads processed chunks (with embeddings) to Supabase.
        """
        print(f"Uploading {len(rows)} rows to Supabase table '{table_name}'...")
        success_count = 0
        failed_count = 0
        
        for i in tqdm(range(0, len(rows), batch_size), desc="Uploading"):
            batch = rows[i:i + batch_size]
            try:
                self.client.table(table_name).insert(batch).execute()
                success_count += len(batch)
            except Exception as e:
                print(f"❌ Batch upload error: {e}")
                # Fallback: insert one by one to isolate the bad row
                for row in batch:
                    try:
                        self.client.table(table_name).insert([row]).execute()
                        success_count += 1
                    except Exception as e2:
                        print(f"❌ Failed row doc_id={row.get('doc_id')}: {e2}")
                        failed_count += 1

        print(f"\n🎉 Upload Complete!")
        print(f"✅ Inserted: {success_count}")
        print(f"❌ Failed: {failed_count}")