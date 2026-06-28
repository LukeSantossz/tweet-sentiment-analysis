use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Instant;
use unicode_segmentation::UnicodeSegmentation;

static URL_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://\S+").unwrap());
static MENTION_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"@\w+").unwrap());
static HASHTAG_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"#(\w+)").unwrap());

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

fn remove_urls(text: &str) -> String {
    URL_PATTERN.replace_all(text, "[URL]").trim().to_string()
}

fn remove_mentions(text: &str) -> String {
    MENTION_PATTERN.replace_all(text, "@user").to_string()
}

fn normalize_hashtags(text: &str) -> String {
    HASHTAG_PATTERN.replace_all(text, "$1").to_string()
}

fn handle_emojis(text: &str) -> String {
    let mut result = String::with_capacity(text.len() * 2);

    // Use grapheme clusters to correctly handle multi-codepoint emojis
    // (flags, skin tones, ZWJ sequences like family emojis)
    for grapheme in text.graphemes(true) {
        if let Some(emoji) = emojis::get(grapheme) {
            result.push(':');
            result.push_str(&emoji.name().replace(' ', "_"));
            result.push(':');
        } else {
            result.push_str(grapheme);
        }
    }
    result
}

fn to_lowercase(text: &str) -> String {
    text.to_lowercase()
}

fn clean_tweet_text(text: &str) -> String {
    let text = remove_urls(text);
    let text = remove_mentions(&text);
    let text = normalize_hashtags(&text);
    let text = handle_emojis(&text);
    to_lowercase(&text)
}

fn process_tweets_parallel(texts: &[String], pb: &ProgressBar) -> Vec<String> {
    texts
        .par_iter()
        .map(|text| {
            pb.inc(1);
            clean_tweet_text(text)
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

    #[test]
    fn test_remove_urls() {
        assert_eq!(
            remove_urls("Here is the link https://site.com to see"),
            "Here is the link [URL] to see"
        );
        assert_eq!(remove_urls("https://site.com"), "[URL]");
    }

    #[test]
    fn test_remove_mentions() {
        assert_eq!(
            remove_mentions("Hey @joao, all good?"),
            "Hey @user, all good?"
        );
        assert_eq!(
            remove_mentions("Happy new year @ana and @carlos!"),
            "Happy new year @user and @user!"
        );
    }

    #[test]
    fn test_normalize_hashtags() {
        assert_eq!(
            normalize_hashtags("I love programming in #python"),
            "I love programming in python"
        );
        assert_eq!(
            normalize_hashtags("I love programming in #python3"),
            "I love programming in python3"
        );
    }

    #[test]
    fn test_to_lowercase() {
        assert_eq!(to_lowercase("Hello World"), "hello world");
        // Unicode-aware lowercasing: non-ASCII letters must lowercase too.
        assert_eq!(to_lowercase("CAFÉ"), "café");
    }

    #[test]
    fn test_handle_emojis() {
        let result = handle_emojis("I am happy 😊");
        assert!(result.contains(":smiling_face_with_smiling_eyes:"));
    }

    #[test]
    fn test_handle_emojis_multi_codepoint() {
        // Test flag emoji (2 regional indicator symbols). The input has no literal
        // "Brazil", so the assertion below only passes if the emoji was converted.
        let result_flag = handle_emojis("Long live 🇧🇷!");
        assert!(
            result_flag.contains("Brazil") && result_flag.contains(":"),
            "Flag should be converted to text: {}",
            result_flag
        );
        // Verify the flag emoji itself is not in output
        assert!(
            !result_flag.contains("🇧🇷"),
            "Flag emoji should be removed: {}",
            result_flag
        );

        // Test skin tone modifier
        let result_skin = handle_emojis("Thumbs up 👍🏻");
        assert!(
            !result_skin.contains("👍"),
            "Skin tone emoji should be converted: {}",
            result_skin
        );

        // Test ZWJ sequence (family emoji)
        let result_family = handle_emojis("Family 👨‍👩‍👧");
        assert!(
            !result_family.contains("👨")
                && !result_family.contains("👩")
                && !result_family.contains("👧"),
            "ZWJ family should be converted: {}",
            result_family
        );
    }

    #[test]
    fn test_clean_tweet_text() {
        let result = clean_tweet_text(
            "Hey @joao, all good? I love programming in #python 😊 https://site.com",
        );
        assert!(result.contains("@user"));
        assert!(result.contains("python"));
        assert!(result.contains("[url]"));
        assert!(result.contains(":smiling_face_with_smiling_eyes:"));
        assert_eq!(result, result.to_lowercase());
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
