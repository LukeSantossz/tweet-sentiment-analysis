use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "tweet-preprocessor")]
#[command(about = "High-performance tweet preprocessing for sentiment analysis at scale")]
#[command(version)]
struct Args {
    /// Input file path (CSV or Parquet)
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path (Parquet)
    #[arg(short, long)]
    output: PathBuf,

    /// Column name containing tweet text
    #[arg(short, long, default_value = "text")]
    text_column: String,

    /// Number of threads (0 = auto)
    #[arg(short = 'j', long, default_value = "0")]
    threads: usize,
}

/// Normalize text to the model-input contract (mirrors `preprocess_for_model` in
/// `src/preprocessing.py`, ADR 0009): collapse `@mentions` to `@user` and URLs to `http`,
/// while preserving case, hashtags, and emoji. This is the single contract the fine-tuned
/// model was trained on; the bulk `clean_tweet_text` contract is intentionally not used here.
fn preprocess_for_model(text: &str) -> String {
    text.split(' ')
        .map(|token| {
            if token.starts_with('@') && token.chars().count() > 1 {
                "@user".to_string()
            } else if token.starts_with("http") {
                "http".to_string()
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn process_tweets_parallel(texts: &[String], pb: &ProgressBar) -> Vec<String> {
    texts
        .par_iter()
        .map(|text| {
            pb.inc(1);
            preprocess_for_model(text)
        })
        .collect()
}

/// Extracts the text column as owned strings, mapping null / missing values to an
/// empty string.
///
/// Null policy is **empty-string** (not skip or hard error): a missing tweet is
/// treated as empty text. This keeps the output row-aligned with the input (the CLI
/// appends a `text_cleaned` column to the existing frame) and never lets a single
/// null abort a multi-million-row batch. `src/preprocessing.py` assumes non-null
/// `str` input, so this CLI is where the scale-time null policy is defined.
fn extract_text_column(df: &DataFrame, column: &str) -> PolarsResult<Vec<String>> {
    Ok(df
        .column(column)?
        .str()?
        .into_iter()
        .map(|opt| opt.unwrap_or("").to_string())
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }

    println!("Tweet Preprocessor v{}", env!("CARGO_PKG_VERSION"));
    println!("{}", "─".repeat(50));
    println!("Input:  {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!("Column: {}", args.text_column);
    println!(
        "Threads: {}",
        if args.threads == 0 {
            "auto".to_string()
        } else {
            args.threads.to_string()
        }
    );
    println!("{}", "─".repeat(50));

    let start = Instant::now();

    println!("\n[1/4] Reading input file...");
    let df = match args.input.extension().and_then(|e| e.to_str()) {
        Some("csv") => CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(args.input.clone()))?
            .finish()?,
        Some("parquet") => {
            let file = std::fs::File::open(&args.input)?;
            ParquetReader::new(file).finish()?
        }
        _ => return Err("Unsupported input format. Use CSV or Parquet.".into()),
    };

    let row_count = df.height();
    println!("  Loaded {} rows", row_count);

    println!("\n[2/4] Extracting text column...");
    let text_series = extract_text_column(&df, &args.text_column)?;

    println!("\n[3/4] Processing tweets...");
    let pb = ProgressBar::new(row_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")?
            .progress_chars("##-"),
    );

    let cleaned_texts = process_tweets_parallel(&text_series, &pb);
    pb.finish_with_message("done");

    println!("\n[4/4] Writing output...");
    let cleaned_series = Series::new(PlSmallStr::from("text_cleaned"), cleaned_texts);

    let mut output_df = df.clone();
    output_df.with_column(cleaned_series)?;

    let file = std::fs::File::create(&args.output)?;
    ParquetWriter::new(file).finish(&mut output_df)?;

    let elapsed = start.elapsed();
    let throughput = row_count as f64 / elapsed.as_secs_f64();

    println!("\n{}", "=".repeat(50));
    println!("Processing Complete!");
    println!("{}", "=".repeat(50));
    println!("  Rows processed: {}", row_count);
    println!("  Time elapsed:   {:.2?}", elapsed);
    println!("  Throughput:     {:.0} tweets/second", throughput);
    println!("  Output:         {}", args.output.display());
    println!("{}", "=".repeat(50));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Model-input contract (ADR 0009). These mirror
    // tests/test_preprocessing.py::test_preprocess_for_model_* byte-for-byte so the Rust and
    // Python implementations stay in parity.
    #[test]
    fn test_preprocess_for_model_normalizes_mentions_and_urls() {
        assert_eq!(
            preprocess_for_model("Hey @joao check http://x.co now"),
            "Hey @user check http now"
        );
    }

    #[test]
    fn test_preprocess_for_model_preserves_case_hashtags_emoji() {
        assert_eq!(
            preprocess_for_model("I LOVE #Python 😊"),
            "I LOVE #Python 😊"
        );
    }

    #[test]
    fn test_preprocess_for_model_leaves_bare_at_symbol() {
        assert_eq!(preprocess_for_model("email @ me"), "email @ me");
    }

    #[test]
    fn test_preprocess_for_model_is_idempotent() {
        let text = "Hey @user check http #NLP 🔥";
        assert_eq!(
            preprocess_for_model(&preprocess_for_model(text)),
            preprocess_for_model(text)
        );
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("twpp_test_{}_{}", std::process::id(), name))
    }

    #[test]
    fn test_extract_text_column_maps_null_to_empty() {
        let df = polars::df!("text" => &[Some("hello @joe"), None, Some("hi")]).unwrap();
        let texts = extract_text_column(&df, "text").unwrap();
        assert_eq!(
            texts,
            vec!["hello @joe".to_string(), String::new(), "hi".to_string()]
        );
    }

    #[test]
    fn test_null_handling_csv() {
        // Row 2 has a missing text field; it must clean to an empty string.
        let csv = "id,text\n1,hello\n2,\n3,world\n";
        let path = temp_path("null.csv");
        std::fs::write(&path, csv).unwrap();

        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.clone()))
            .unwrap()
            .finish()
            .unwrap();
        let texts = extract_text_column(&df, "text").unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(
            texts,
            vec!["hello".to_string(), String::new(), "world".to_string()]
        );
    }

    #[test]
    fn test_null_handling_parquet() {
        let mut df = polars::df!("text" => &[Some("hello"), None, Some("world")]).unwrap();
        let path = temp_path("null.parquet");

        let file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(file).finish(&mut df).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let df_read = ParquetReader::new(file).finish().unwrap();
        let texts = extract_text_column(&df_read, "text").unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(
            texts,
            vec!["hello".to_string(), String::new(), "world".to_string()]
        );
    }
}
