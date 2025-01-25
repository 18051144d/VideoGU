import numpy as np

## Evaluation code
def calculate_iou(gt, pred):
    """
    Calculate IoU of two segments.
    Args:
        gt: (start, end)
        pred: (start, end)
    Returns:
        iou
    """
    intersection = max(0, min(gt[1], pred[1]) - max(gt[0], pred[0]))
    union = max(gt[1], pred[1]) - min(gt[0], pred[0])
    if union == 0:
        return 0
    else:
        return float(intersection) / union

def calculate_tiou(gt_segments, pred_segments, tiou_thresholds, num_classes = 177):
    """
    Calculate temporal intersection over union (tIoU) for each prediction.
    Args:
        gt_segments: list of ground truth segments, each segment is (start, end, class)
        pred_segments: list of predicted segments, each segment is (start, end, class, score)
        tiou_thresholds: list of tIoU thresholds
        num_classes: number of classes
    Returns:
        tiou_values: list of tIoU values for each prediction
    """
    tiou_values = np.zeros((num_classes, len(pred_segments), len(tiou_thresholds)))

    for c in range(num_classes):
      pred_segments_c = [p for p in pred_segments if p[2] == c]
      gt_segments_c = [g for g in gt_segments if g[2] == c]
      for i, pred_segment in enumerate(pred_segments_c):
          for j, gt_segment in enumerate(gt_segments_c):
              iou = calculate_iou(gt_segment[:2], pred_segment[:2])
              for k, tiou_threshold in enumerate(tiou_thresholds):
                  if iou >= tiou_threshold:
                      tiou_values[c,i, k] = 1
    return tiou_values

def calculate_map(gt_segments_list, pred_segments_list, tiou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9], num_classes = 177, remove_nexist = True):
    """
    Calculate mean average precision (mAP) for temporal action localization.
    Args:
        gt_segments_list: list of ground truth segments for each video
        pred_segments_list: list of predicted segments for each video
        tiou_thresholds: list of tIoU thresholds
        num_classes: number of classes
    Returns:
        map_at_thresholds: mAP at each tIoU threshold
    """
    exist = []
    num_videos = len(gt_segments_list)
    num_thresholds = len(tiou_thresholds)
    ap_at_thresholds = np.zeros((num_classes, num_thresholds))

    for video_idx in range(num_videos):
        gt_segments = gt_segments_list[video_idx]
        pred_segments = sorted(pred_segments_list[video_idx], key=lambda x: x[3], reverse=True)

        tiou_values = calculate_tiou(gt_segments, pred_segments, tiou_thresholds, num_classes)

        for c in range(num_classes):
          for tiou_idx in range(num_thresholds):
              tp = tiou_values[c,:, tiou_idx]
              fp = 1 - tp
              num_gt = len([g for g in gt_segments if g[2] == c])
              
              if num_gt == 0:
                if len([p for p in pred_segments if p[2] == c]) == 0:
                  ap = 1.0
                else:
                  ap = 0.0
              else:
                exist += [c]
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(num_gt)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

                ap = 0.0
                for t in np.arange(0.0, 1.1, 0.1):
                    if np.sum(rec >= t) == 0:
                        p = 0
                    else:
                        p = np.max(prec[rec >= t])
                    ap = ap + p / 11.0

              ap_at_thresholds[c, tiou_idx] += ap

    if remove_nexist:
        ap_at_thresholds = ap_at_thresholds[exist]

    map_at_thresholds = ap_at_thresholds / num_videos
    map_at_thresholds = np.mean(map_at_thresholds, axis=0)

    return map_at_thresholds

## Output parsing
def fast_parsing(str_):
    parsed = ''
    for s in str_:
        if s not in [str(i) for i in range(10)] + [',', ' ']:
            continue
        parsed += s
    return [int(i) for i in eval(f'[{parsed}]')]


## TAL code
def temporal_action_localization(classification_results, separation_threshold = 7, minimum_action_time = 7):
    """
    Performs temporal action localization on a list of classification results.

    Args:
        classification_results: A list of dictionaries, each representing the classification result of a frame.
                                 Each dictionary has the format: {time_id: int, cls: int}.
        separation_threshold: The maximum time difference between two frames to be considered part of the same action.
        minimum_action_time: The minimum duration for an action to be considered valid.

    Returns:
        A list of dictionaries, each representing a temporal action.
        Each dictionary has the format: {start: int, end: int, cls: int}.
    """

    if not classification_results:
        return []

    classification_results.sort(key=lambda x: x['time_id'])
    
    temporal_actions = []
    current_action = None
    
    for i in range(len(classification_results)):
        result = classification_results[i]

        if current_action is None:
            current_action = {'start': result['time_id'], 'end': result['time_id'], 'cls': result['cls'], 'indices': [i]}
        else:
            if result['cls'] == current_action['cls'] and result['time_id'] - classification_results[current_action['indices'][-1]]['time_id'] <= separation_threshold:
                current_action['end'] = result['time_id']
                current_action['indices'].append(i)
            elif result['time_id'] - classification_results[current_action['indices'][-1]]['time_id'] <= separation_threshold:
                
                
                
                
                found_same_class = False
                for j in reversed(current_action['indices']):
                    if classification_results[j]['cls'] == result['cls']:
                        current_action['end'] = result['time_id']
                        current_action['indices'].append(i)
                        found_same_class = True
                        break
                if not found_same_class:
                    if current_action['end'] - current_action['start'] >= minimum_action_time:
                        temporal_actions.append({'start': current_action['start'], 'end': current_action['end'], 'cls': current_action['cls']})
                    current_action = {'start': result['time_id'], 'end': result['time_id'], 'cls': result['cls'], 'indices': [i]}

            else:
                if current_action['end'] - current_action['start'] >= minimum_action_time:
                    temporal_actions.append({'start': current_action['start'], 'end': current_action['end'], 'cls': current_action['cls']})
                current_action = {'start': result['time_id'], 'end': result['time_id'], 'cls': result['cls'], 'indices': [i]}

    if current_action is not None and current_action['end'] - current_action['start'] >= minimum_action_time:
        temporal_actions.append({'start': current_action['start'], 'end': current_action['end'], 'cls': current_action['cls']})

    return temporal_actions