use std::sync::Arc;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_array::types::Float32Type;
use fastembed::{Embedding, TextEmbedding};
use lancedb::{connect, Connection, Table};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::arrow::IntoArrow;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use thiserror::Error;
use futures::TryStreamExt;
use log::warn;
use serde::Deserialize;


const EMBED_TABLE_NAME: &str = "note_embeddings";
const EMBEDDING_DIMENSIONS: i32 = 384;
struct EmbeddingsDb {
    vec_db: Connection,
    embedding_engine: TextEmbedding,
}


// Define the Embeddable trait
trait Embeddable {
    fn id(&self) -> &String;
    fn text(&self) -> &String;
}

// Define the Document struct
#[derive(Deserialize)]
struct Document {
    id: String,
    text: String,
}

// Implement the Embeddable trait for Document
impl Embeddable for Document {
    fn id(&self) -> &String {
        &self.id
    }

    fn text(&self) -> &String {
        &self.text
    }
}



impl EmbeddingsDb {

    pub async fn new(db_path: &str, embedding_engine: TextEmbedding) -> Result<EmbeddingsDb, lancedb::error::Error> {
        let db_conn = connect(db_path).execute().await?;
        let embed_db = EmbeddingsDb {
            vec_db: db_conn,
            embedding_engine
        };
        embed_db.init_table(EMBED_TABLE_NAME).await?;

        Ok(embed_db)

    }

    pub(crate) async fn get_table_names(&self) -> Result<Vec<String>,  lancedb::error::Error> {
        Ok(self.vec_db.table_names().execute().await?)
    }

    async fn init_table(&self, table_name: &str) -> Result<(), lancedb::error::Error> {
        let table_names = self.vec_db.table_names().execute().await?;
        let table_exists = table_names.contains(&table_name.to_string());
        if !table_exists {
            let schema = self.get_table_schema();
            self.vec_db.create_empty_table(table_name, schema).execute().await?;
        }
        Ok(())
    }

    fn get_table_schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "embeddings",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_DIMENSIONS,
                ),
                true,
            )
        ]))
    }

    pub(crate) async fn add_documents<T: Embeddable>(&self, data: Vec<T>) -> Result<(), EmbedDbError> {
        let ids: Vec<String> = data.iter().map(|doc| doc.id().to_string()).collect();
        let texts: Vec<String> = data.iter().map(|doc| doc.text().to_string()).collect();
        assert_eq!(ids.len(), texts.len(), "The length of all data attribute vectors must all be the same");
        let embeddings = self.create_embeddings(&texts)?;
        assert_eq!(
            embeddings[0].len() as i32,
            EMBEDDING_DIMENSIONS,
            "Embedding dimensions mismatch"
        );
        log::info!("Saving Documents: {:?}", ids);
        // get record batch stream
        let records_batch = self.get_record_batch_stream(
            self.get_table_schema(), ids, texts, embeddings).await?;
        let table = self.vec_db.open_table(EMBED_TABLE_NAME).execute().await?;
        table.add(records_batch).execute().await?;
        Ok(())
    }

    pub(crate) async fn empty(&self) -> Result<(), EmbedDbError> {
        self.vec_db.drop_table(EMBED_TABLE_NAME).await?;
        self.init_table(EMBED_TABLE_NAME).await?;
        Ok(())
    }

    pub(crate) async fn get_document_by_id(&self, id: &str) -> Result<Vec<Document>, EmbedDbError> {
        let filter = format!("id = '{}'", id);
        let table = self.vec_db.open_table(EMBED_TABLE_NAME).execute().await?;
        let result = table.query()
            .only_if(filter)
            // no need to return embeddings
            .select(Select::Columns(vec!["id".to_string(), "text".to_string()]))
            .execute().await?
            .try_collect::<Vec<_>>().await?;
        if result.len() > 1 {
            warn!("Greater than one record returned for id {}. Found {} total", id, result.len());
        }
        let mut documents: Vec<Document> = Vec::new();
        for item in result {
            let x: Vec<Document> = serde_arrow::from_record_batch(&item)?;
            documents.extend(x);
        }
        Ok(documents)
    }

    pub(crate) async fn create_index(table: &Table) -> lancedb::Result<()> {
        table.create_index(&["vector"], Index::Auto).execute().await
    }

    pub(crate) async fn documents_count(&self) -> Result<usize, EmbedDbError> {
        let table = self.vec_db.open_table(EMBED_TABLE_NAME).execute().await?;
        Ok(table.count_rows(None).await?)
    }

    async fn get_record_batch_stream(&self, schema: SchemaRef, ids:Vec<String>, texts: Vec<String>,
                                     embeddings: Vec<Vec<f32>>) -> lancedb::Result<impl IntoArrow> {
        // Below, from_iter_primitive creates a FixedSizeListArray from an iterator of primitive values
        // Example
        // let data = vec![
        //    Some(vec![Some(0), Some(1), Some(2)]),
        //    None,
        //    Some(vec![Some(3), None, Some(5)]),
        //    Some(vec![Some(6), Some(7), Some(45)]),
        // ];
        // let list_array = FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(data, 3);
        // println!("{:?}", list_array);
        //
        // Thus we wrap all the embeddings items in Options
        let option_wrapped_embeddings: Vec<_> = embeddings.into_iter()
            .map(|vec| {
                Some(vec.into_iter().map(Some).collect::<Vec<_>>())
            })
            .collect();
        let batches = RecordBatchIterator::new(
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Arc::new(StringArray::from(ids)) as ArrayRef),
                    Arc::new(Arc::new(StringArray::from(texts)) as ArrayRef),
                    Arc::new(
                        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                            option_wrapped_embeddings,
                            EMBEDDING_DIMENSIONS,
                        ),
                    ),
                ],
            )
                .unwrap()]
                .into_iter()
                .map(Ok),
            schema.clone(),
        );
        Ok(Box::new(batches))
    }







    pub fn storage_path(&self) -> String {
        self.vec_db.uri().to_string()
    }
    
    pub fn create_embeddings(&self, documents: &[String]) -> Result<Vec<Embedding>, EmbedDbError> {
        self.embedding_engine
            .embed(documents.to_vec(), None)
            .map_err(EmbedDbError::from)
    }
}

#[derive(Error, Debug)]
pub enum EmbedDbError {
    #[error("Embedding error: {0}")]
    Embedding(#[from] anyhow::Error),
    #[error("LanceDb error: {0}")]
    LanceDb(#[from] lancedb::Error),
    #[error("SerDe error: {0}")]
    SerDe(#[from] serde_arrow::Error),

}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use fastembed::{EmbeddingModel, InitOptions};
    use super::*;
    use tempfile::TempDir;

    fn remove_dir_if_exists<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
        if path.as_ref().exists() {
            fs::remove_dir_all(path)
        } else {
            Ok(())
        }
    }

    async fn get_embed_db() -> (EmbeddingsDb, String) {
        let test_db_path = "test_db";
        // remove test_db dir if it exists
        remove_dir_if_exists(test_db_path);
        let embedding_engine = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::AllMiniLML6V2,
            show_download_progress: true,
            ..Default::default()
        }).unwrap();
        (EmbeddingsDb::new(test_db_path, embedding_engine).await.unwrap(),
         test_db_path.to_string())
    }
    
    #[tokio::test]
    async fn test_suite() {
        // not able to overcome some sort of race-condition/state-issue with construction
        // of TextEmbedding per test.  Also, not successful in implementing some sort of singleton
        // pattern over TextEmbedding (use a single instance for all tests)
        // So... I'm left with this, <shrug>, it works and I have a test suite. 
        let (embed_db, embed_db_path)  = get_embed_db().await;
        create_embed_db(&embed_db, &embed_db_path).await;
        embed_db.empty().await.unwrap();
        create_embed_db_table(&embed_db).await;
        embed_db.empty().await.unwrap();
        create_embeddings(&embed_db).await;
        embed_db.empty().await.unwrap();
        test_add_documents(&embed_db).await;
        embed_db.empty().await.unwrap();
        test_empty_db(&embed_db).await;
    }
    
    async fn create_embed_db(embed_db: &EmbeddingsDb, embed_db_path: &str) {
        assert_eq!(embed_db.storage_path(), embed_db_path);
    }


    async fn create_embed_db_table(embed_db: &EmbeddingsDb) {
        let table_names = embed_db.get_table_names().await.unwrap();
        assert_eq!(table_names.len(), 1);
        assert_eq!(table_names.first().unwrap(), EMBED_TABLE_NAME);
    }
    
    async fn create_embeddings(embed_db: &EmbeddingsDb) {
        let data = vec!["hello world".to_string()];
        let embeddings = embed_db.create_embeddings(&data).unwrap();
        assert_eq!(embeddings.len(), data.len(), "The returned item is one vec per given data item");
        assert_eq!(embeddings[0].len(), 384, "The embeddings within the returned vec should be 384 floats (AllMiniLML6V2 uses 384 dimensions)");
    }

    async fn test_add_documents(embed_db: &EmbeddingsDb) {
        let documents: Vec<Document> = vec![
            Document {
                id: "1".to_string(),
                text: "Hello world".to_string(),
            },
            Document {
                id: "2".to_string(),
                text: "Rust programming".to_string(),
            },
        ];
        embed_db.add_documents(documents).await.unwrap();
        assert_eq!(embed_db.documents_count().await.unwrap(), 2,
                   "Just added 2 docs so expecting 2 from table count");
        let record_1 = embed_db.get_document_by_id("1").await.unwrap().pop();
        assert!(record_1.is_some());
        assert_eq!("Hello world", record_1.unwrap().text )
    }
    
    async fn test_empty_db(embed_db: &EmbeddingsDb) {
        let documents: Vec<Document> = vec![
            Document {
                id: "1".to_string(),
                text: "Hello world".to_string(),
            },
            Document {
                id: "2".to_string(),
                text: "Rust programming".to_string(),
            },
        ];
        embed_db.add_documents(documents).await.unwrap();
        assert_eq!(embed_db.documents_count().await.unwrap(), 2,
                   "Just added 2 docs so expecting 2 from table count");
        embed_db.empty().await;
        let count = embed_db.documents_count().await.unwrap();
        assert_eq!(count, 0);
        
    }



}
