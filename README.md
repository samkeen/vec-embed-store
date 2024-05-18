# VecEmbedStore

This is a thin wrapper around LanceDb (VectorDb) meant to provide a means to create/store/query
embeddings in a LanceDb without the need to grok the lower level Arrow/ColumnarDb tech.

# Usage example

```rust
fn main() {

    // Instance EmbeddingsDb
    let db_dir_path = "db";
    let embedding_engine_options = EmbeddingEngineOptions {
        model_name: EmbeddingModel::BGESmallENV15,
        ..Default::default()
    };
    let embed_db = EmbeddingsDb::new(db_dir_path, embedding_engine_options).await.unwrap();

    // Add documents
    let documents = vec![
        TextBlock {
            id: "1".to_string(),
            text: "Hello world".to_string(),
        },
        TextBlock {
            id: "2".to_string(),
            text: "Rust programming".to_string(),
        },
    ];
    embed_db.add_texts(texts).await.unwrap();

    // check for similarities for a given document
    let search_doc = TextBlock {
        id: "3".to_string(),
        text: "Hello world".to_string(),
    };
    let result: Vec<ComparedTextBlock> = embed_db
        .get_similar_to(search_doc.clone())
        //.threshold(0.001) // Optional minimum distance for vector distance comparisons
        //.limit(10)        // Optional limit number items returned
        .execute()
        .await
        .unwrap();

    dbg!(result)
    // result = [
    //     ComparedTextBlock {
    //         id: "1",
    //         text: "Hello world",
    //         distance: 0.0,
    //     },
    //     ComparedTextBlock {
    //         id: "2",
    //         text: "Rust programming",
    //         distance: 0.66691905,
    //     },
    // ]
}
```

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
