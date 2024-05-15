//! A library for storing and searching text embeddings using a vector database.
//!
//! This library provides an `EmbeddingsDb` struct for managing a database of text embeddings,
//! allowing efficient storage, retrieval, and similarity search.
//!
//! The key components of the library include:
//! - `EmbeddingsDb`: The main struct for interacting with the embeddings database.
//! - `SimilaritySearch`: A builder-style struct for performing similarity searches on the embeddings.
//! - `EmbedText`: A struct representing a text to be embedded and stored in the database.
//! - `ComparedEmbedText`: A struct representing a text with its similarity distance after a search.
//! - `EmbeddingEngineOptions`: A struct for configuring the embedding engine options.
use std::path::{Path, PathBuf};
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use fastembed::{Embedding, EmbeddingModel, ExecutionProviderDispatch, InitOptions, TextEmbedding};
use futures::TryStreamExt;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::arrow::IntoArrow;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{connect, Connection, Table};
use log::warn;
use serde::Deserialize;
use std::sync::Arc;
use thiserror::Error;

const EMBED_TABLE_NAME: &str = "note_embeddings";
const EMBEDDING_DIMENSIONS: i32 = 384;
const DEFAULT_EMBEDDINGS_CACHE_DIR: &str = ".fastembed_cache";

/// The main struct for interacting with the embeddings database.
struct EmbeddingsDb {
    vec_db: Connection,
    embedding_engine: TextEmbedding,
}

/// A builder-style struct for performing similarity searches on the embeddings.
pub struct SimilaritySearch<'a> {
    embed_db: &'a EmbeddingsDb,
    embed_text: TextBlock,
    threshold: Option<f32>,
    limit: Option<usize>,
}

impl<'a> SimilaritySearch<'a> {
    /// Creates a new `SimilaritySearch` instance.
    pub fn new(embed_db: &'a EmbeddingsDb, embed_text: TextBlock) -> Self {
        SimilaritySearch {
            embed_db,
            embed_text,
            threshold: None,
            limit: None,
        }
    }

    /// Sets the similarity threshold for the search.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Sets the maximum number of results to return.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Executes the similarity search and returns the results.
    pub async fn execute(self) -> Result<Vec<ComparedTextBlock>, EmbedDbError> {
        let embedding = self.embed_db.create_embeddings(&[self.embed_text.text])?;
        // flattening a 2D vector into a 1D vector. This is necessary because the search
        // function of the Table trait expects a 1D vector as input. However, the
        // create_embeddings function returns a 2D vector (a vector of embeddings,
        // where each embedding is itself a vector)
        let embedding: Vec<f32> = embedding
            .into_iter()
            .flat_map(|embedding| embedding.to_vec())
            .collect();
        let table = self
            .embed_db
            .vec_db
            .open_table(EMBED_TABLE_NAME)
            .execute()
            .await?;
        let query = table
            .query()
            .select(Select::Columns(vec!["id".to_string(), "text".to_string()]))
            .nearest_to(embedding)?;

        let query = if let Some(limit) = self.limit {
            query.limit(limit)
        } else {
            query
        };
        let result = query.execute().await?.try_collect::<Vec<_>>().await?;
        convert_to_compared_embed_texts(result, &self.threshold)
    }
}

/// A struct representing a text to be embedded and stored in the database.
#[derive(Deserialize, Clone, Debug)]
pub struct TextBlock {
    pub id: String,
    pub text: String,
}

/// A struct representing a text with its similarity distance after a search.
#[derive(Deserialize, Clone, Debug)]
pub struct ComparedTextBlock {
    pub id: String,
    pub text: String,
    #[serde(rename = "_distance")]
    pub distance: f32,
}

/// A struct for configuring the embedding engine options.
#[derive(Debug, Clone)]
pub struct EmbeddingEngineOptions {
    // pub model_name: EmbeddingModel,
    // pub execution_providers: Vec<ExecutionProviderDispatch>,
    // pub max_length: usize,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl Default for EmbeddingEngineOptions {
    fn default() -> Self {
        Self {
            // model_name: DEFAULT_EMBEDDING_MODEL,
            // execution_providers: Default::default(),
            // max_length: DEFAULT_MAX_LENGTH,
            cache_dir: Path::new(DEFAULT_EMBEDDINGS_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
        }
    }
}

impl EmbeddingsDb {
    /// Creates a new instance of `EmbeddingsDb`.
    pub async fn new(
        db_path: &str,
        embedding_engine_options: EmbeddingEngineOptions
    ) -> Result<EmbeddingsDb, EmbedDbError> {
        let embedding_engine = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::AllMiniLML6V2,
            show_download_progress: embedding_engine_options.show_download_progress,
            cache_dir: embedding_engine_options.cache_dir,
            ..Default::default()
        })?;
        let db_conn = connect(db_path).execute().await?;
        let embed_db = EmbeddingsDb {
            vec_db: db_conn,
            embedding_engine,
        };
        embed_db.init_table(EMBED_TABLE_NAME).await?;

        Ok(embed_db)
    }

    /// Retrieves the names of all tables in the database.
    pub async fn get_table_names(&self) -> Result<Vec<String>, EmbedDbError> {
        self.vec_db.table_names().execute().await.map_err(EmbedDbError::LanceDb)
    }

    /// Initializes the embeddings table if it doesn't exist.
    async fn init_table(&self, table_name: &str) -> Result<(), EmbedDbError> {
        let table_names = self.vec_db.table_names().execute().await?;
        let table_exists = table_names.contains(&table_name.to_string());
        if !table_exists {
            let schema = self.get_table_schema();
            self.vec_db
                .create_empty_table(table_name, schema)
                .execute()
                .await?;
        }
        Ok(())
    }

    /// Retrieves the schema for the embeddings table.
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
            ),
        ]))
    }

    /// Adds a batch of texts to the embeddings database.
    pub async fn add_texts(&self, data: Vec<TextBlock>) -> Result<(), EmbedDbError> {
        let ids: Vec<String> = data.iter().map(|doc| doc.id.to_string()).collect();
        let texts: Vec<String> = data.iter().map(|doc| doc.text.to_string()).collect();
        assert_eq!(
            ids.len(),
            texts.len(),
            "The length of all data attribute vectors must all be the same"
        );
        let embeddings = self.create_embeddings(&texts)?;
        assert_eq!(
            embeddings[0].len() as i32,
            EMBEDDING_DIMENSIONS,
            "Embedding dimensions mismatch"
        );
        log::info!("Saving Texts: {:?}", ids);
        // get record batch stream
        let records_batch = self
            .get_record_batch_stream(self.get_table_schema(), ids, texts, embeddings)
            .await?;
        let table = self.vec_db.open_table(EMBED_TABLE_NAME).execute().await?;
        table.add(records_batch).execute().await?;
        Ok(())
    }

    /// Clears all data from the embeddings database.
    pub async fn empty_db(&self) -> Result<(), EmbedDbError> {
        self.vec_db.drop_table(EMBED_TABLE_NAME).await?;
        self.init_table(EMBED_TABLE_NAME).await?;
        Ok(())
    }

    /// Retrieves a text from the database by its ID.
    pub async fn get_text_by_id(
        &self,
        id: &str,
    ) -> Result<Vec<TextBlock>, EmbedDbError> {
        let filter = format!("id = '{}'", id);
        let table = self.vec_db.open_table(EMBED_TABLE_NAME).execute().await?;
        let result = table
            .query()
            .only_if(filter)
            // no need to return embeddings
            .select(Select::Columns(vec!["id".to_string(), "text".to_string()]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        if result.len() > 1 {
            warn!(
                "Greater than one record returned for id {}. Found {} total",
                id,
                result.len()
            );
        }
        convert_to_embed_texts(result)
    }

    /// Creates a new `SimilaritySearch` instance for finding similar texts.
    pub fn get_similar_to(&self, embed_text: TextBlock) -> SimilaritySearch {
        SimilaritySearch::new(self, embed_text)
    }

    /// Creates an index on the embeddings table.
    pub async fn create_index(table: &Table) -> lancedb::Result<()> {
        table.create_index(&["vector"], Index::Auto).execute().await
    }

    /// Retrieves the total number of items in the embeddings database.
    pub async fn items_count(&self) -> Result<usize, EmbedDbError> {
        let table = self.vec_db.open_table(EMBED_TABLE_NAME).execute().await?;
        Ok(table.count_rows(None).await?)
    }

    /// Generates a record batch stream from the provided data.
    async fn get_record_batch_stream(
        &self,
        schema: SchemaRef,
        ids: Vec<String>,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
    ) -> lancedb::Result<impl IntoArrow> {
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
        let option_wrapped_embeddings: Vec<_> = embeddings
            .into_iter()
            .map(|vec| Some(vec.into_iter().map(Some).collect::<Vec<_>>()))
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

    /// Retrieves the storage path of the embeddings database.
    pub(crate) fn storage_path(&self) -> String {
        self.vec_db.uri().to_string()
    }

    /// Creates embeddings for the given texts using the embedding engine.
    pub(crate) fn create_embeddings(&self, texts: &[String]) -> Result<Vec<Embedding>, EmbedDbError> {
        self.embedding_engine
            .embed(texts.to_vec(), None)
            .map_err(EmbedDbError::from)
    }
}

/// Converts the record batch result to a vector of `EmbedText` instances.
fn convert_to_embed_texts(result: Vec<RecordBatch>) -> Result<Vec<TextBlock>, EmbedDbError> {
    let mut texts: Vec<TextBlock> = Vec::new();
    for item in result {
        let x: Vec<TextBlock> = serde_arrow::from_record_batch(&item)?;
        texts.extend(x);
    }
    Ok(texts)
}

/// Converts the record batch result to a vector of `ComparedEmbedText` instances,
/// filtering based on the provided threshold.
fn convert_to_compared_embed_texts(
    result: Vec<RecordBatch>,
    threshold: &Option<f32>,
) -> Result<Vec<ComparedTextBlock>, EmbedDbError> {
    let mut compared_embed_texts: Vec<ComparedTextBlock> = Vec::new();
    for item in result {
        let x: Vec<ComparedTextBlock> = serde_arrow::from_record_batch(&item)?;
        if let Some(threshold_value) = threshold {
            compared_embed_texts.extend(x.into_iter().filter(|doc| &doc.distance <= threshold_value));
        } else {
            compared_embed_texts.extend(x);
        }
    }
    Ok(compared_embed_texts)
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
    use super::*;
    use std::fs;
    use std::path::Path;

    fn remove_dir_if_exists<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
        if path.as_ref().exists() {
            fs::remove_dir_all(path)
        } else {
            Ok(())
        }
    }

    async fn get_embed_db() -> (EmbeddingsDb, String) {
        let test_db_path = "test_db";
        remove_dir_if_exists(test_db_path).expect("Failed removing test db dir");
        let embedding_engine_options = EmbeddingEngineOptions {
            ..Default::default()
        };
        (
            EmbeddingsDb::new(test_db_path, embedding_engine_options)
                .await
                .unwrap(),
            test_db_path.to_string(),
        )
    }

    fn get_texts() -> Vec<TextBlock> {
        vec![
            TextBlock {
                id: "1".to_string(),
                text: "Hello world".to_string(),
            },
            TextBlock {
                id: "2".to_string(),
                text: "Rust programming".to_string(),
            },
        ]
    }

    #[tokio::test]
    async fn test_suite() {
        // not able to overcome some sort of race-condition/state-issue with construction
        // of TextEmbedding per test.  Also, not successful in implementing some sort of singleton
        // pattern over TextEmbedding (use a single instance for all tests)
        // So... I'm left with this, <shrug>, it works and I have a test suite.
        let (embed_db, embed_db_path) = get_embed_db().await;
        create_embed_db(&embed_db, &embed_db_path).await;
        embed_db.empty_db().await.unwrap();
        create_embed_db_table(&embed_db).await;
        embed_db.empty_db().await.unwrap();
        create_embeddings(&embed_db).await;
        embed_db.empty_db().await.unwrap();
        test_add_texts(&embed_db).await;
        embed_db.empty_db().await.unwrap();
        test_empty_db(&embed_db).await;
        embed_db.empty_db().await.unwrap();
        test_get_similar_texts(&embed_db).await;
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
        assert_eq!(
            embeddings.len(),
            data.len(),
            "The returned item is one vec per given data item"
        );
        assert_eq!(embeddings[0].len() as i32, EMBEDDING_DIMENSIONS, "The embeddings within the returned vec should be 384 floats (AllMiniLML6V2 uses 384 dimensions)");
    }

    async fn test_add_texts(embed_db: &EmbeddingsDb) {
        let texts = get_texts();
        embed_db.add_texts(texts).await.unwrap();
        assert_eq!(
            embed_db.items_count().await.unwrap(),
            2,
            "Just added 2 docs so expecting 2 from table count"
        );
        let record_1 = embed_db.get_text_by_id("1").await.unwrap().pop();
        assert!(record_1.is_some());
        assert_eq!("Hello world", record_1.unwrap().text)
    }

    async fn test_empty_db(embed_db: &EmbeddingsDb) {
        let texts = get_texts();
        embed_db.add_texts(texts).await.unwrap();
        assert_eq!(
            embed_db.items_count().await.unwrap(),
            2,
            "Just added 2 docs so expecting 2 from table count"
        );
        embed_db.empty_db().await.unwrap();
        let count = embed_db.items_count().await.unwrap();
        assert_eq!(count, 0);
    }

    async fn test_get_similar_texts(embed_db: &EmbeddingsDb) {
        let texts = get_texts();
        embed_db.add_texts(texts).await.unwrap();
        let search_doc = TextBlock {
            id: "3".to_string(),
            text: "Hello world".to_string(),
        };
        let result = embed_db
            .get_similar_to(search_doc.clone())
            .execute()
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            2,
            "No limit so we should see both docs returned"
        );
        assert_eq!(result[0].id, "1");
        assert_eq!(result[1].id, "2");

        let result = embed_db
            .get_similar_to(search_doc.clone())
            .limit(1)
            .execute()
            .await
            .unwrap();
        assert_eq!(result.len(), 1, "limit so we should only 1 doc returned");
        assert_eq!(
            result[0].id, "1",
            "The compare doc and doc 1 share the same text so it should return"
        );

        let result = embed_db
            .get_similar_to(search_doc.clone())
            .threshold(0.001)
            .execute()
            .await
            .unwrap();
        assert_eq!(
            result.len(),
            1,
            "very small threshold, so we should only 1 doc returned"
        );
        assert_eq!(
            result[0].id, "1",
            "The compare doc and doc 1 share the same text so it should return"
        );
        assert_eq!(
            result[0].distance, 0.0,
            "the docs are identical so distance should be 0"
        )
    }
}
