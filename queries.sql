-- Q1. Summary per rank: how does quality improve as k grows
SELECT rank_k,
       ROUND(AVG(mse), 6)      AS avg_mse,
       ROUND(AVG(loss_pct), 2) AS avg_loss_pct,
       ROUND(MIN(loss_pct), 2) AS best_case_loss,
       ROUND(MAX(loss_pct), 2) AS worst_case_loss
FROM rank_evaluations
GROUP BY rank_k
ORDER BY rank_k;

-- Q2. Hardest people to rebuild at k = 30 (AVG + HAVING, 25 percent worse than average)
SELECT m.subject_id,
       ROUND(AVG(e.loss_pct), 2) AS avg_loss_pct,
       COUNT(*)                  AS n_photos
FROM rank_evaluations e
JOIN image_metadata m ON m.image_id = e.image_id
WHERE e.rank_k = 30
GROUP BY m.subject_id
HAVING AVG(e.loss_pct) > (SELECT AVG(loss_pct) FROM rank_evaluations WHERE rank_k = 30) * 1.25
ORDER BY avg_loss_pct DESC
LIMIT 10;

-- Q3. Quality buckets: Lossy (20 percent loss or more), Standard (5 to 20), Near-Lossless (under 5)
WITH bucketed AS (
    SELECT rank_k,
           CASE
               WHEN loss_pct >= 20 THEN 'Lossy'
               WHEN loss_pct >= 5  THEN 'Standard'
               ELSE 'Near-Lossless'
           END AS quality
    FROM rank_evaluations
)
SELECT rank_k,
       SUM(quality = 'Lossy')         AS lossy,
       SUM(quality = 'Standard')      AS standard,
       SUM(quality = 'Near-Lossless') AS near_lossless
FROM bucketed
GROUP BY rank_k
ORDER BY rank_k;

-- Q4. Best and worst photo of each person at k = 30 (RANK window function)
WITH photo_ranked AS (
    SELECT m.subject_id,
           m.photo_index,
           ROUND(e.loss_pct, 2) AS loss_pct,
           RANK() OVER (PARTITION BY m.subject_id ORDER BY e.loss_pct ASC)  AS best_first,
           RANK() OVER (PARTITION BY m.subject_id ORDER BY e.loss_pct DESC) AS worst_first
    FROM rank_evaluations e
    JOIN image_metadata m ON m.image_id = e.image_id
    WHERE e.rank_k = 30
)
SELECT subject_id, photo_index, loss_pct,
       CASE WHEN best_first = 1 THEN 'best photo' ELSE 'worst photo' END AS which_photo
FROM photo_ranked
WHERE best_first = 1 OR worst_first = 1
ORDER BY subject_id, loss_pct;

-- Q5. First rank where each person's average loss drops to 10 percent or less (ROW_NUMBER)
WITH subject_rank AS (
    SELECT m.subject_id, e.rank_k, AVG(e.loss_pct) AS avg_loss
    FROM rank_evaluations e
    JOIN image_metadata m ON m.image_id = e.image_id
    GROUP BY m.subject_id, e.rank_k
),
acceptable AS (
    SELECT subject_id, rank_k, ROUND(avg_loss, 2) AS avg_loss,
           ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY rank_k ASC) AS nth
    FROM subject_rank
    WHERE avg_loss <= 10
)
SELECT subject_id, rank_k AS first_acceptable_k, avg_loss
FROM acceptable
WHERE nth = 1
ORDER BY first_acceptable_k DESC, subject_id;

-- Q6. Diminishing returns: gain from each extra step in rank (LAG window function)
WITH per_rank AS (
    SELECT rank_k, AVG(loss_pct) AS avg_loss
    FROM rank_evaluations
    GROUP BY rank_k
)
SELECT rank_k,
       ROUND(avg_loss, 2) AS avg_loss,
       ROUND(LAG(avg_loss) OVER (ORDER BY rank_k) - avg_loss, 2) AS gain_vs_previous_rank,
       ROUND((LAG(avg_loss) OVER (ORDER BY rank_k) - avg_loss)
             / (rank_k - LAG(rank_k) OVER (ORDER BY rank_k)), 3) AS gain_per_extra_component
FROM per_rank
ORDER BY rank_k;

-- Q7. Faces that look unusual to the model: Mahalanobis d2 above 37.57 at k = 20
SELECT m.subject_id, m.photo_index,
       ROUND(e.mahalanobis_d2, 1) AS d2,
       ROUND(e.loss_pct, 1)       AS loss_pct
FROM rank_evaluations e
JOIN image_metadata m ON m.image_id = e.image_id
WHERE e.rank_k = 20 AND e.mahalanobis_d2 > 37.57
ORDER BY e.mahalanobis_d2 DESC;

-- Q8. Most inconsistent people at k = 30 (spread = worst photo minus best photo)
SELECT m.subject_id,
       ROUND(MIN(e.loss_pct), 2)                   AS best_photo_loss,
       ROUND(MAX(e.loss_pct), 2)                   AS worst_photo_loss,
       ROUND(MAX(e.loss_pct) - MIN(e.loss_pct), 2) AS spread
FROM rank_evaluations e
JOIN image_metadata m ON m.image_id = e.image_id
WHERE e.rank_k = 30
GROUP BY m.subject_id
ORDER BY spread DESC
LIMIT 5;