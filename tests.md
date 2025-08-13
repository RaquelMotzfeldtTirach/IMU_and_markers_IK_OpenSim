
# Sensor Fusion Test Results

## Test Configuration Table

| Test ID | Subject |    Movement    | Marker Weight | Orientation Weight | Constraint | Marker RMS (m) | Marker Max (m) | Orientation RMS (°) | Orientation Max (°) | Total Cost | Notes |
|---------|---------|----------------|---------------|-------------------|------------|----------------|----------------|-------------------|-------------------|------------|-------|
| T001    | 28      | elbow          | 5 (4200)      | 1 (4200)         | ∞          | 0.096731       | 0.168809       | 27.1342           | 46.1183           | 1.90651091 | Baseline test |
| T002    | 28      | elbow          | 500 (550)     | 1 (2)            | ∞          | 0.148785       | 0.240879       | 19.1773           | 32.0427           | 1.16187037 | High marker weight |
| T003    | 28      | elbow          | 10 (15)       | 100 (90)         | ∞          | 0.153548       | 0.241208       | 18.6804           | 31.4824           | 1.13331274 | Balanced weights |
| T004    | 28      | elbow          | 0             | 1 (2)            | ∞          | -              | -              | 18.7258           | 27.3961           | 0.85452184 | IMU only |
| T005    | 28      | elbow          | 1 (2)         | 0                | ∞          | 0.024137       | 0.050328       | -                 | -                 | 0.00699106 | Markers only |
| T006    | 28      | shoulder_abd   | 1 (2)         | 0                | ∞          | 0.027243       | 0.057977       | -                 | -                 | 0.00890620 | Markers only |
| T007    | 28      | shoulder_abd   | 0             | 1 (2)            | ∞          | -              | -              | 19.9219           | 30.8378           | 0.96717960 | IMU Only   |
| T008    | 28      | shoulder_abd   | 1 (2)         | 1 (2)            | ∞          | 0.103415       | 0.188008       | 20.1399           | 30.0942           | 1.11679673 | Balanced   |
| T009    | 28      | shoulder_abd   | 10 (50)       | 100 (200)        | ∞          | 0.106070       | 0.159501       | 20.2099           | 29.9424           | 1.13035759 | High orientation |
| T010    |         |                |               |                  |            |                |                |                   |                   |            |            |

### Legend
- **Weight Format**: `base_weight (last_element_weight)` - The first number is the standard weight, the number in parentheses is the weight for the last marker/orientation
- **Constraint**: ∞ = Infinite weight, numbers = finite constraint weight
- **RMS**: Root Mean Square error
- **Max**: Maximum error across all elements
- **Total Cost**: Combined cost function value

### Analysis Notes

#### Best Configurations
- **Lowest Total Cost**: T005 (markers only) - 0.00699106
- **Best Balanced**: T003 (10/100 weights) - 1.13331274
- **Lowest Marker Error**: T005 (markers only) - 0.024137m RMS
- **Lowest Orientation Error**: T004 (IMU only) - 18.7258° RMS

#### Key Observations
1. **Markers-only mode** (T005) achieves excellent marker tracking (0.024m RMS) but loses orientation constraints
2. **IMU-only mode** (T004) provides reasonable orientation tracking (~18.7°) but no marker constraints
3. **Combined modes** show trade-offs between marker and orientation accuracy
4. **High marker weights** (T002) improve orientation tracking at the cost of marker accuracy
5. **Balanced weights** (T003) provide the best compromise for combined sensor fusion

---

## Detailed Test Results

### Original Test Results
=== FINAL ERROR SUMMARY ===
Markers: 12 tracked
  - Final RMS error: 0.096731 m
  - Final max error: 0.168809 m
  - Total squared error: 0.11228252
Orientations: 8 tracked
  - Final RMS error: 27.1342°
  - Final max error: 46.1183°
  - Total squared error: 1.79422839
Total final cost: 1.90651091
============================

500 (550), 1(2), infinity
=== FINAL ERROR SUMMARY ===
Markers: 12 tracked
Marker weights: 
  - Final RMS error: 0.148785 m
  - Final max error: 0.240879 m
  - Total squared error: 0.26564209
Orientations: 8 tracked
Orientation weights:
  - Final RMS error: 19.1773°
  - Final max error: 32.0427°
  - Total squared error: 0.89622828
Total final cost: 1.16187037
=================================
10(15), 100(90), infinity
=== FINAL ERROR SUMMARY ===
Markers: 12 tracked
  - Final RMS error: 0.153548 m
  - Final max error: 0.241208 m
  - Total squared error: 0.28292234
Orientations: 8 tracked
  - Final RMS error: 18.6804°
  - Final max error: 31.4824°
  - Total squared error: 0.85039040
Total final cost: 1.13331274
=====================================
only orientation 1 (2)
=== FINAL ERROR SUMMARY ===
Orientations: 8 tracked
  - Final RMS error: 18.7258°
  - Final max error: 27.3961°
  - Total squared error: 0.85452184
Total final cost: 0.85452184
=============================
only markers 1 (2)
=== FINAL ERROR SUMMARY ===
Markers: 12 tracked
  - Final RMS error: 0.024137 m
  - Final max error: 0.050328 m
  - Total squared error: 0.00699106
Total final cost: 0.00699106
========================================




