# SQL exact-match results (500 held-out examples)

| Model | Path | Accuracy | Correct | vs base | Usable as-is |
|---|---|---|---|---|---|
| Qwen3.8-27B base | `/mnt/data/qwen38-demo/models/Qwen3.8-27B` | 55.4% | 277/500 | - | 1.6% |
| LoRA (1 epoch) | `/mnt/data/qwen38-demo/output/qwen3.8-27b-sql` | 87.4% | 437/500 | +32.0pp | 86.8% |
| Full fine-tune (1 epoch) | `/mnt/data/qwen38-demo/output/qwen3.8-27b-sql-full` | 86.6% | 433/500 | +31.2pp | 86.0% |

**Accuracy** extracts the SQL from the response (markdown fence or last `SELECT`) and normalises quote style, case and whitespace before comparing. **Usable as-is** is the same comparison with no extraction and no quote handling — a diagnostic for whether the output can be used without post-processing, not a measure of SQL correctness.

## Examples LoRA (1 epoch) fixed (167 total)

### Example 5

**Question:** What is the name of the player who is Sco and moving to greenock morton in the summer?

**Ground truth:** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

**Qwen3.8-27B base:** ````sql
SELECT name FROM table_name_83 WHERE name = 'Sco' AND moving_to = 'greenock morton' AND transfer_window = 'summer';
````

**LoRA (1 epoch):** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

### Example 6

**Question:** Of all the contestants who got voted, what is the contestant number and name of the contestant who got least votes?

**Ground truth:** `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

**Qwen3.8-27B base:** ````sql
SELECT c.contestant_number, c.contestant_name
FROM contestants c
JOIN votes v ON c.contestant_number = v.contestant_number
GROUP BY c.contestant_number, c.contestant_name
ORDER BY COUNT(v.contestant_number) ASC
LIMIT 1;
````

**LoRA (1 epoch):** `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

### Example 8

**Question:** What did the tournament that got an A in 1945 get in 1949?

**Ground truth:** `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

**Qwen3.8-27B base:** ````sql
SELECT t2.grade
FROM table_name_22 t1
JOIN table_name_22 t2 ON t1.Id = t2.Id
WHERE t1.grade = 'A' AND t1.year = 1945 AND t2.year = 1949;
````

**LoRA (1 epoch):** `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

### Example 9

**Question:** Find the states where have the colleges whose enrollments are less than the largest size.

**Ground truth:** `SELECT DISTINCT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`

**Qwen3.8-27B base:** ````sql
SELECT state
FROM college
WHERE enr < (SELECT MAX(enr) FROM college);
````

**LoRA (1 epoch):** `SELECT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`

### Example 16

**Question:** Who is the opponent of the game played on November 20, 1995?

**Ground truth:** `SELECT opponent FROM table_name_66 WHERE date = "november 20, 1995"`

**Qwen3.8-27B base:** `SELECT opponent FROM table_name_66 WHERE date = '1995-11-20'`

**LoRA (1 epoch):** `SELECT opponent FROM table_name_66 WHERE date = "november 20, 1995"`

## Examples Full fine-tune (1 epoch) fixed (162 total)

### Example 5

**Question:** What is the name of the player who is Sco and moving to greenock morton in the summer?

**Ground truth:** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

**Qwen3.8-27B base:** ````sql
SELECT name FROM table_name_83 WHERE name = 'Sco' AND moving_to = 'greenock morton' AND transfer_window = 'summer';
````

**Full fine-tune (1 epoch):** `SELECT name FROM table_name_83 WHERE nat = "sco" AND transfer_window = "summer" AND moving_to = "greenock morton"`

### Example 6

**Question:** Of all the contestants who got voted, what is the contestant number and name of the contestant who got least votes?

**Ground truth:** `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

**Qwen3.8-27B base:** ````sql
SELECT c.contestant_number, c.contestant_name
FROM contestants c
JOIN votes v ON c.contestant_number = v.contestant_number
GROUP BY c.contestant_number, c.contestant_name
ORDER BY COUNT(v.contestant_number) ASC
LIMIT 1;
````

**Full fine-tune (1 epoch):** `SELECT T1.contestant_number, T1.contestant_name FROM contestants AS T1 JOIN votes AS T2 ON T1.contestant_number = T2.contestant_number GROUP BY T1.contestant_number ORDER BY COUNT(*) LIMIT 1`

### Example 8

**Question:** What did the tournament that got an A in 1945 get in 1949?

**Ground truth:** `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

**Qwen3.8-27B base:** ````sql
SELECT t2.grade
FROM table_name_22 t1
JOIN table_name_22 t2 ON t1.Id = t2.Id
WHERE t1.grade = 'A' AND t1.year = 1945 AND t2.year = 1949;
````

**Full fine-tune (1 epoch):** `SELECT 1949 FROM table_name_22 WHERE 1945 = "a"`

### Example 9

**Question:** Find the states where have the colleges whose enrollments are less than the largest size.

**Ground truth:** `SELECT DISTINCT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`

**Qwen3.8-27B base:** ````sql
SELECT state
FROM college
WHERE enr < (SELECT MAX(enr) FROM college);
````

**Full fine-tune (1 epoch):** `SELECT state FROM college WHERE enr < (SELECT MAX(enr) FROM college)`

### Example 16

**Question:** Who is the opponent of the game played on November 20, 1995?

**Ground truth:** `SELECT opponent FROM table_name_66 WHERE date = "november 20, 1995"`

**Qwen3.8-27B base:** `SELECT opponent FROM table_name_66 WHERE date = '1995-11-20'`

**Full fine-tune (1 epoch):** `SELECT opponent FROM table_name_66 WHERE date = "november 20, 1995"`

