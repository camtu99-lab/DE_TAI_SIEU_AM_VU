# Báo cáo thực nghiệm BreastVision-XAI V3 (full_experiment)

**Full experiment. Kết quả dưới đây là số liệu thực từ lần chạy này; chưa phải kết luận lâm sàng.**

Thiết bị: cuda; IMG_SIZE=256; EPOCHS=40; seeds=[42, 43, 44]; pretrained=True

## Dữ liệu
- BUS-BRA: 1875 ảnh, 1064 bệnh nhân; split theo bệnh nhân: {'test': 272, 'train': 1336, 'val': 267}
- Rò rỉ: {'group_across_splits': 0, 'patient_id_across_splits': 0, 'sha256_across_splits': 0, 'neardup_pairs_across_splits': 0, 'ok': True}
- BUSI (ngoài, nhị phân): 645 ảnh; Normal bị loại; bản sao với BUS-BRA bị loại: 0

## Nội bộ (BUS-BRA test): trung bình ± sd qua seed
```
    model  n_seeds  auroc_mean  auroc_sd  auprc_mean  auprc_sd  sensitivity_malignant_mean  sensitivity_malignant_sd  specificity_mean  specificity_sd  balanced_accuracy_mean  balanced_accuracy_sd  f1_mean  f1_sd  dice_mean  dice_sd  iou_mean  iou_sd  hd95_mean  hd95_sd
       E0        1      0.9185       NaN      0.8298       NaN                      0.8767                       NaN            0.7286             NaN                  0.8027                   NaN   0.6702    NaN        NaN      NaN       NaN     NaN        NaN      NaN
       E3        1      0.8883       NaN      0.7823       NaN                      0.7945                       NaN            0.7940             NaN                  0.7942                   NaN   0.6744    NaN     0.8743      NaN    0.7954     NaN    13.1523      NaN
  V3_base        1      0.9380       NaN      0.8565       NaN                      0.9041                       NaN            0.7839             NaN                  0.8440                   NaN   0.7253    NaN     0.8959      NaN    0.8250     NaN     8.7515      NaN
V3_birads        1      0.9397       NaN      0.8588       NaN                      0.9178                       NaN            0.8040             NaN                  0.8609                   NaN   0.7486    NaN     0.9080      NaN    0.8392     NaN     7.7303      NaN
  V3_full        3      0.9195    0.0105      0.8397    0.0108                      0.9041                    0.0237            0.7588          0.0174                  0.8315                0.0179   0.7060 0.0215     0.8978   0.0052    0.8261  0.0054     8.6922   0.2684
       X0        3      0.9197    0.0111      0.8389    0.0182                      0.8493                    0.0548            0.8241          0.0611                  0.8367                0.0131   0.7308 0.0271     0.8861   0.0047    0.8087  0.0072    10.9228   1.1405
       X3        3      0.9108    0.0033      0.8156    0.0063                      0.8584                    0.0481            0.7672          0.0865                  0.8128                0.0242   0.6916 0.0415     0.8947   0.0112    0.8220  0.0136     9.6145   2.0906
       X4        3      0.9227    0.0034      0.8138    0.0097                      0.8813                    0.0554            0.7806          0.0556                  0.8309                0.0011   0.7117 0.0150     0.8976   0.0048    0.8261  0.0067     9.0649   0.6205
       X6        3      0.9296    0.0119      0.8400    0.0191                      0.8630                    0.0362            0.8425          0.0636                  0.8528                0.0207   0.7555 0.0456     0.8973   0.0050    0.8278  0.0069     8.8929   0.8257
```

## So sánh cặp (paired bootstrap)
```
        A         B metric    diff  ci_low  ci_high                                 verdict
       X4        X3  auroc  0.0113 -0.0132   0.0400 CI chứa 0: chưa đủ bằng chứng khác biệt
       X4        X3   dice -0.0077 -0.0177   0.0007 CI chứa 0: chưa đủ bằng chứng khác biệt
       X6        X3  auroc  0.0222 -0.0001   0.0470 CI chứa 0: chưa đủ bằng chứng khác biệt
       X6        X3   dice  0.0001 -0.0086   0.0081 CI chứa 0: chưa đủ bằng chứng khác biệt
       X6        X0  auroc  0.0093 -0.0120   0.0326 CI chứa 0: chưa đủ bằng chứng khác biệt
       X6        X0   dice  0.0135  0.0022   0.0245                     CI > 0: A cao hơn B
       X6        E3  auroc  0.0425  0.0048   0.0801                     CI > 0: A cao hơn B
       X6        E3   dice  0.0254  0.0128   0.0394                     CI > 0: A cao hơn B
  V3_base        X6  auroc  0.0068 -0.0066   0.0210 CI chứa 0: chưa đủ bằng chứng khác biệt
  V3_base        X6   dice -0.0038 -0.0091   0.0016 CI chứa 0: chưa đủ bằng chứng khác biệt
V3_birads   V3_base  auroc  0.0015 -0.0109   0.0141 CI chứa 0: chưa đủ bằng chứng khác biệt
V3_birads   V3_base   dice  0.0119  0.0048   0.0207                     CI > 0: A cao hơn B
  V3_full V3_birads  auroc -0.0078 -0.0243   0.0074 CI chứa 0: chưa đủ bằng chứng khác biệt
  V3_full V3_birads   dice -0.0042 -0.0081  -0.0002                     CI < 0: B cao hơn A
  V3_full        X6  auroc  0.0005 -0.0172   0.0180 CI chứa 0: chưa đủ bằng chứng khác biệt
  V3_full        X6   dice  0.0039 -0.0034   0.0134 CI chứa 0: chưa đủ bằng chứng khác biệt
  V3_full        X0  auroc  0.0098 -0.0111   0.0314 CI chứa 0: chưa đủ bằng chứng khác biệt
  V3_full        X0   dice  0.0174  0.0096   0.0265                     CI > 0: A cao hơn B
```

## XAI (localisation / faithfulness / consistency / collapse)
```
                  group                                        metric V3_birads   V3_full        X0        X3        X4        X6
background_perturbation evidence_mass_in_roi_after_background_shuffle  0.751774  0.852282       NaN  0.253093  0.842556  0.845963
background_perturbation                   evidence_mass_in_roi_before  0.785338  0.806663       NaN  0.073333  0.965196  0.772382
background_perturbation             mean_abs_dprob_background_shuffle  0.283135  0.281131  0.517432  0.352529  0.190863  0.214914
background_perturbation                 mean_abs_dprob_lesion_shuffle  0.172329   0.17587  0.184837  0.197614  0.198237  0.212458
background_perturbation                                             n     100.0     100.0     100.0     100.0     100.0     100.0
               collapse                                 collapse_flag       0.0       0.0       NaN       0.0       0.0       0.0
               collapse                             mean_ess_fraction  0.117302  0.106091       NaN  0.117286  0.007555  0.119121
               collapse                             mean_norm_entropy  0.721669  0.706327       NaN  0.748731  0.302325   0.72636
               collapse                                             n     272.0     272.0       NaN     272.0     272.0     272.0
               collapse                          peaked_collapse_rate       0.0       0.0       NaN       0.0  0.058824       0.0
               collapse                         uniform_collapse_rate       0.0       0.0       NaN       0.0       0.0       0.0
     consistency_affine            evidence_cosine_after_inverse_warp  0.995463   0.99424       NaN  0.573923   0.50802  0.996421
     consistency_affine                      mean_abs_dprob_malignant  0.089116  0.092133  0.105597  0.119948  0.125362  0.087743
     consistency_affine                                             n     100.0     100.0     100.0     100.0     100.0     100.0
     consistency_affine                          prediction_agreement      0.92       0.9      0.89      0.88      0.88      0.89
      consistency_hflip            evidence_cosine_after_inverse_warp  0.996406  0.994273       NaN  0.564291  0.289974  0.997267
      consistency_hflip                      mean_abs_dprob_malignant  0.098362  0.066446  0.087798  0.137745  0.085335  0.087561
      consistency_hflip                                             n     100.0     100.0     100.0     100.0     100.0     100.0
      consistency_hflip                          prediction_agreement      0.89      0.94       0.9      0.84      0.93      0.93
  evidence_intervention                                    applicable       1.0       1.0       0.0       1.0       1.0       1.0
  evidence_intervention                              mean_drop_random  0.007593  0.008402       NaN -0.000177  0.005675   0.00258
  evidence_intervention                        mean_drop_top_evidence  0.332226  0.372619       NaN  0.000427  0.051311   0.01716
  evidence_intervention                                             n     100.0     100.0       NaN     100.0     100.0     100.0
  evidence_intervention                                             q       0.2       0.2       NaN       0.2       0.2       0.2
           faithfulness                         deletion_auc_evidence  0.599467  0.526015       NaN  0.728906  0.766211   0.70768
           faithfulness                          deletion_auc_gradcam  0.436458  0.570848  0.360977  0.558043  0.620416  0.570163
           faithfulness                           deletion_auc_random  0.610209  0.623933  0.447551  0.705496  0.664762  0.749287
           faithfulness                        insertion_auc_evidence   0.76849  0.765018       NaN   0.63413  0.721562  0.815954
           faithfulness                         insertion_auc_gradcam  0.796608  0.676992  0.653227  0.837259  0.821795  0.827058
           faithfulness                          insertion_auc_random  0.601424  0.630224  0.445841   0.72008  0.665413    0.7337
           faithfulness                                             n     100.0     100.0     100.0     100.0     100.0     100.0
  localisation_evidence                          area_adjusted_lesion  0.545656  0.581847       NaN -0.048883  0.799129  0.558384
  localisation_evidence                  area_adjusted_lesion_context   0.74428  0.777334       NaN -0.060331  0.947003  0.749608
  localisation_evidence                                             n     272.0     272.0       NaN     272.0     272.0     272.0
  localisation_evidence                                 pointing_game       1.0       1.0       NaN  0.051471  0.974265  0.996324
  localisation_evidence                         relevance_mass_lesion  0.567763  0.601725       NaN  0.023899  0.808543  0.580277
  localisation_evidence                 relevance_mass_lesion_context  0.764567  0.794486       NaN   0.05833  0.951107  0.769789
   localisation_gradcam                          area_adjusted_lesion  0.074557   0.06195  0.249615  0.180609  0.167021  0.131793
   localisation_gradcam                  area_adjusted_lesion_context  0.112605  0.076243  0.370353  0.261661  0.240031  0.177528
   localisation_gradcam                                             n     100.0     100.0     100.0     100.0     100.0     100.0
   localisation_gradcam                                 pointing_game      0.01      0.24      0.66      0.72      0.68      0.72
   localisation_gradcam                         relevance_mass_lesion  0.134386  0.122977  0.295707  0.232199  0.220332  0.186939
   localisation_gradcam                 relevance_mass_lesion_context   0.20673   0.17475   0.43361  0.338329  0.320301  0.263567
```

## Hiệu chuẩn và chọn lọc dự đoán
```
  model  seed hiệu_chuẩn      T    NLL  Brier    ECE
V3_full    44      trước 1.0000 0.3689 0.2139 0.0495
V3_full    44        sau 1.2592 0.3567 0.2112 0.0260
     X0    43      trước 1.0000 0.7325 0.2322 0.1105
     X0    43        sau 5.7372 0.2970 0.1860 0.0226
     X6    43      trước 1.0000 0.7798 0.2144 0.1049
     X6    43        sau 5.5292 0.2983 0.1802 0.0377
  model   aurc  coverage_target  coverage_actual  selective_risk  abstain_threshold  sensitivity_malignant_accepted  malignant_abstained
V3_full 0.0704              0.9           0.8971          0.1107             0.6932                          0.8154               0.1096
V3_full 0.0704              0.8           0.7831          0.0798             0.8189                          0.8596               0.2192
V3_full 0.0704              0.7           0.7206          0.0765             0.8676                          0.8846               0.2877
     X0 0.0331              0.9           0.9044          0.1016             0.6211                          0.7705               0.1644
     X0 0.0331              0.8           0.7574          0.0631             0.7482                          0.8085               0.3562
     X0 0.0331              0.7           0.6434          0.0286             0.8292                          0.8649               0.4932
     X6 0.0363              0.9           0.8971          0.0943             0.6308                          0.8923               0.1096
     X6 0.0363              0.8           0.7941          0.0602             0.7646                          0.8966               0.2055
     X6 0.0363              0.7           0.7096          0.0415             0.8406                          0.9400               0.3151
```

## Đánh giá ngoài BUSI (pipeline đã khóa)
- X0: AUROC 0.8662 (95% CI 0.833-0.896), AUPRC 0.8056, sens 0.856, spec 0.615, ECE 0.031
- X6: AUROC 0.8864 (95% CI 0.856-0.911), AUPRC 0.8168, sens 0.813, spec 0.782, ECE 0.043
- V3_full: AUROC 0.8807 (95% CI 0.846-0.908), AUPRC 0.8378, sens 0.895, spec 0.587, ECE 0.037

## Giới hạn
- BUS-BRA một quốc gia; BUSI không có mã bệnh nhân; chưa kiểm định lâm sàng.
- Nếu CI của hiệu số chứa 0 thì chưa đủ bằng chứng khác biệt; không viết "vượt trội".
- Localisation ≠ consistency ≠ faithfulness: chỉ faithfulness (deletion/insertion/intervention) nói về ảnh hưởng nhân quả tới dự đoán.
- Ứng dụng là công cụ hỗ trợ nghiên cứu, không phải thiết bị chẩn đoán.