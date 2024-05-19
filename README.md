# VecEmbedStore

This is a thin wrapper around LanceDb (VectorDb) meant to provide a means to create/store/query
embeddings in a LanceDb without the need to grok the lower level Arrow/ColumnarDb tech.

## Usage example

Add VecEmbedStore to dependencies

```bash
cargo add vec_embed_store
# If you want to select a Embedding engine other than the default, you currently need to add fastembed 
# This is an issue open to remove this requirement: https://github.com/samkeen/vec-embed-store/issues/9
# [optional] 
cargo add fastembed

```

```rust
use std::path::PathBuf;
use vec_embed_store::{EmbeddingsDb, EmbeddingEngineOptions, TextChunk, SimilaritySearch};
use fastembed::EmbeddingModel::BGESmallENV15;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up the embedding engine options, 
    let embedding_engine_options = EmbeddingEngineOptions {
        model_name: BGESmallENV15, // see https://docs.rs/fastembed/latest/fastembed/enum.EmbeddingModel.html
        cache_dir: PathBuf::from("path/to/cache"),
        show_download_progress: true,
        ..Default::default()
    };

    // Create a new instance of EmbeddingsDb
    let embed_db = EmbeddingsDb::new("path/to/db", embedding_engine_options).await?;

    // Define the texts to be added to the database.  Chunk texts in any way you see fit (and matches with the 
    //   chosen embedding engine).
    let texts = vec![
        TextChunk {
            id: "1".to_string(),
            text: "Once upon a midnight dreary, while I pondered, weak and weary,".to_string(),
        },
        TextChunk {
            id: "2".to_string(),
            text: "Over many a quaint and curious volume of forgotten lore—".to_string(),
        },
        TextChunk {
            id: "3".to_string(),
            text: "While I nodded, nearly napping, suddenly there came a tapping,".to_string(),
        },
    ];

    // Upsert the texts into the embeddings database (there is no separate add/update)
    // TextChunks MUST be unique on `TextChunk.id`
    embed_db.upsert_texts(&texts).await?;

    // Retrieve a text by its ID
    let retrieved_text = embed_db.get_text_by_id("1").await?;
    println!("Retrieved text: {:?}", retrieved_text);

    // Define a text for similarity search
    let search_text = "suddenly there came a tapping";

    // Perform a similarity search
    let search_results = embed_db
        .get_similar_to(search_text)
        .limit(2)
        .threshold(0.8)
        .execute()
        .await?;

    println!("Similarity search results:");
    for result in search_results {
        println!("ID: {}, Text: {}, Distance: {}", result.id, result.text, result.distance);
    }

    // Get all text chunks from the database
    let all_texts = embed_db.get_all_texts().await?;
    println!("All texts: {:?}", all_texts);

    // Delete texts by their IDs
    let ids_to_delete = vec!["2".to_string(), "3".to_string()];
    embed_db.delete_texts(&ids_to_delete).await?;

    // Get the count of items in the database
    let count = embed_db.items_count().await?;
    println!("Number of items in the database: {}", count);

    // Clear all data from the database
    embed_db.empty_db().await?;

    Ok(())
}
```

## Architecture

EmbedStore encapsulates the Embedding engine and the VectorDb, providing a
simple interface to store and query text blocks.
Currently,  [FastEmbed-rs](https://github.com/Anush008/fastembed-rs) is used for embeddings
and [LanceDb](https://lancedb.github.io/lancedb/) is used for the vector Db

```text

    +----------------------------------------------------------+
    |                      VecEmbedStore                       |
    |                                                          |
    |  +-------------------+           +--------------+        |
    |  | EmbeddingEngine   |           |  VectorDB    |        |
    |  +-------------------+           +--------------+        |
    |                                                          |
    +----------------------------------------------------------+
               ^                              |
               | store                        | similarity search
               |                              v
    +--------------+                   +-----------------------+
    |  TextBlock   |                   | ComparedTextBlock     |
    +--------------+                   +-----------------------+
    
    
```
