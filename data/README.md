AVS GT

The file avs_gt_visione_mapping.csv contains the AVS queries and jusdgment  runned in trecvid for 2019 to 2023. It contains teh folowing columns:

- query: id of the AVS task

- query_text: text of the AVS task

- videoID: id of a video

-shotID: id of the video shot (original V3C name)

-visioneShotID: id of a visione keyframe in the reference shot. note that there is not a 1-1 correspondence between shotsID and  visioneShotID (we may have extracted mre frame from a single shot)

-collection:  is the subset V3c1 or v3c3

-judgment: -1 not in judged sample, 1 shot contains concept,  0 shot doesn't contain concept

-sampling_stratum: (1(top),2,3(i do not know teh meaning, it cames from the trecvid GT)

-start : start6 time of the video shot (using trecvid interval)

-end: ent time of the video shot (using trecvid interval)

-year: the year of teh trecvid GT

-The judgment is -1 not in judged sample, 1 shot contains concept,  0 shot doesn't contain concept

In the file avs_gt_visione_mapping_query_judgment.csv there are a counting for each queries of how many data images were annotated.