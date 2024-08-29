mod common;

use std::collections::HashSet;

use candystore::{CandyStore, Config, Result};

use crate::common::{run_in_tempdir, LONG_VAL};

#[test]
fn test_logic() -> Result<()> {
    run_in_tempdir(|dir| {
        let db = CandyStore::open(
            dir,
            Config {
                max_shard_size: 20 * 1024, // use small files to force lots of splits and compactions
                min_compaction_threashold: 10 * 1024,
                ..Default::default()
            },
        )?;

        assert!(db.get("my name")?.is_none());
        db.set("my_name", "inigo montoya")?;
        db.set("your_name", "dread pirate robert")?;

        assert!(db.contains("my_name")?);
        assert!(!db.contains("My NaMe")?);

        assert_eq!(db.get("my_name")?, Some("inigo montoya".into()));
        assert_eq!(db.get("your_name")?, Some("dread pirate robert".into()));
        db.set("your_name", "vizzini")?;
        assert_eq!(db.get("your_name")?, Some("vizzini".into()));
        assert_eq!(db.remove("my_name")?, Some("inigo montoya".into()));
        assert!(db.remove("my_name")?.is_none());
        assert!(db.get("my name")?.is_none());

        let stats = db.stats();
        assert_eq!(stats.num_entries(), 1);
        assert_eq!(stats.num_compactions, 0);
        assert_eq!(stats.num_splits, 0);
        println!("{stats}");

        for _ in 0..1000 {
            db.set(
                "a very long keyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
                LONG_VAL,
            )?;
            assert!(db
                .remove("a very long keyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")?
                .is_some());
        }

        let stats1 = db.stats();
        println!("{stats1}");
        assert_eq!(stats1.num_entries(), 1);
        assert!(stats1.num_compactions >= 2);
        //assert_eq!(stats1.num_splits, 0);

        for i in 0..1000 {
            db.set(&format!("unique key {i}"), LONG_VAL)?;
        }

        let stats2 = db.stats();
        assert_eq!(stats2.num_entries(), 1001);
        assert!(stats2.num_splits > stats1.num_splits);

        assert_eq!(db.get("your_name")?, Some("vizzini".into()));
        db.clear()?;
        assert_eq!(db.get("your_name")?, None);

        let stats3 = db.stats();
        assert_eq!(stats3.num_entries(), 0);
        assert_eq!(stats3.num_compactions, 0);
        assert_eq!(stats3.num_splits, 0);

        for i in 0..1000 {
            db.set(&format!("unique key {i}"), LONG_VAL)?;
        }

        let mut all_keys = HashSet::new();

        for res in db.iter() {
            let (key, val) = res?;
            assert_eq!(val, LONG_VAL.as_bytes());
            assert!(key.starts_with(b"unique key "));
            all_keys.insert(key);
        }

        assert_eq!(all_keys.len(), 1000);

        all_keys.clear();

        let cookie = {
            let mut iter1 = db.iter();
            for _ in 0..100 {
                let res = iter1.next().unwrap();
                let (key, _) = res?;
                all_keys.insert(key);
            }
            iter1.cookie()
        };

        for res in db.iter_from_cookie(cookie) {
            let (key, _) = res?;
            all_keys.insert(key);
        }

        assert_eq!(all_keys.len(), 1000);

        Ok(())
    })
}

#[test]
fn test_histogram() -> Result<()> {
    run_in_tempdir(|dir| {
        let db = CandyStore::open(
            dir,
            Config {
                expected_number_of_keys: 100_000, // pre-split
                ..Default::default()
            },
        )?;

        db.set("k1", "bbb")?;
        db.set("k2", &vec![b'b'; 100])?;
        db.set("k3", &vec![b'b'; 500])?;
        db.set("k4", &vec![b'b'; 5000])?;
        db.set("k4", &vec![b'b'; 4500])?;
        db.set("k5", &vec![b'b'; 50000])?;
        db.set("kkkkkkkkkkkkkkk", &vec![b'b'; 0xffff])?;

        let stats = db.stats();
        assert_eq!(stats.entries_under_128, 2);
        assert_eq!(stats.entries_under_1k, 1);
        assert_eq!(stats.entries_under_8k, 2);
        assert_eq!(stats.entries_over_32k, 2);

        Ok(())
    })
}
