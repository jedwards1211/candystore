mod common;

use candystore::{CandyStore, Config, Result};

use crate::common::run_in_tempdir;

#[test]
fn test_kaki() -> Result<()> {
    run_in_tempdir(|dir| {
        let db = CandyStore::open(
            dir,
            Config {
                max_shard_size: 1024 * 1024,
                min_compaction_threashold: 32 * 1024,
                ..Default::default()
            },
        )?;

        for i in 0..1000 {
            db.set(&format!("key"), &vec![i as u8; 200])?;
        }
        db.wait_for_compaction()?;

        println!(
            "{:?}",
            std::fs::read_dir(dir)
                .unwrap()
                .map(|e| {
                    let e = e.unwrap();
                    (e.file_name(), e.metadata().unwrap().len())
                })
                .collect::<Vec<_>>()
        );

        Ok(())
    })
}
