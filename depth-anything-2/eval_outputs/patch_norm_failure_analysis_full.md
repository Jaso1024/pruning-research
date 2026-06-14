# Patch-Norm Input Pruning Failure Analysis

- Images analyzed: 1033/1033
- Dense accuracy: 0.9516 (1968/2068)
- Readout layers: [2, 5, 8, 11]

## Accuracy and Transitions

| keep | kept patches | accuracy | regressions | fixes | stable correct | stable wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 1356 | 0.9507 | 4 | 2 | 1964 | 98 |
| 0.98 | 1342 | 0.9521 | 3 | 4 | 1965 | 96 |
| 0.95 | 1301 | 0.9492 | 13 | 8 | 1955 | 92 |
| 0.9 | 1233 | 0.9405 | 36 | 13 | 1932 | 87 |
| 0.85 | 1164 | 0.9342 | 56 | 20 | 1912 | 80 |
| 0.8 | 1096 | 0.9270 | 77 | 26 | 1891 | 74 |

## Regression Attribution

| keep | group | n | endpoint deleted | top nonself readout deleted | readout deleted attention | last-layer deleted attention |
|---:|---|---:|---:|---:|---:|---:|
| 0.99 | regression | 4 | 0.0000 | 0.0000 | 0.0146 | 0.0152 |
| 0.99 | stable_correct | 1964 | 0.0183 | 0.0331 | 0.0080 | 0.0094 |
| 0.99 | fix | 2 | 0.0000 | 0.0000 | 0.0050 | 0.0104 |
| 0.99 | stable_wrong | 98 | 0.0102 | 0.0000 | 0.0056 | 0.0074 |
| 0.98 | regression | 3 | 0.0000 | 0.0000 | 0.0307 | 0.0208 |
| 0.98 | stable_correct | 1965 | 0.0336 | 0.0621 | 0.0152 | 0.0177 |
| 0.98 | fix | 4 | 0.0000 | 0.0000 | 0.0104 | 0.0146 |
| 0.98 | stable_wrong | 96 | 0.0104 | 0.0208 | 0.0103 | 0.0138 |
| 0.95 | regression | 13 | 0.6154 | 0.4615 | 0.1508 | 0.1441 |
| 0.95 | stable_correct | 1955 | 0.0639 | 0.1212 | 0.0340 | 0.0399 |
| 0.95 | fix | 8 | 0.0000 | 0.0000 | 0.0230 | 0.0324 |
| 0.95 | stable_wrong | 92 | 0.0217 | 0.0543 | 0.0242 | 0.0321 |
| 0.9 | regression | 36 | 0.6111 | 0.7500 | 0.2041 | 0.1900 |
| 0.9 | stable_correct | 1932 | 0.1071 | 0.2008 | 0.0628 | 0.0742 |
| 0.9 | fix | 13 | 0.1538 | 0.1538 | 0.0757 | 0.0921 |
| 0.9 | stable_wrong | 87 | 0.0690 | 0.0920 | 0.0439 | 0.0569 |
| 0.85 | regression | 56 | 0.6964 | 0.7857 | 0.2513 | 0.2361 |
| 0.85 | stable_correct | 1912 | 0.1532 | 0.2563 | 0.0900 | 0.1061 |
| 0.85 | fix | 20 | 0.1500 | 0.2000 | 0.0953 | 0.1200 |
| 0.85 | stable_wrong | 80 | 0.1000 | 0.1875 | 0.0653 | 0.0822 |
| 0.8 | regression | 77 | 0.6883 | 0.7532 | 0.3013 | 0.2870 |
| 0.8 | stable_correct | 1891 | 0.1920 | 0.3231 | 0.1179 | 0.1381 |
| 0.8 | fix | 26 | 0.2308 | 0.4615 | 0.1299 | 0.1533 |
| 0.8 | stable_wrong | 74 | 0.0946 | 0.1892 | 0.0852 | 0.1070 |

## Global Deleted Attention

Mean dense patch-to-patch attention mass landing on tokens deleted by the threshold.

| keep | layer 0 | layer 2 | layer 5 | layer 8 | layer 11 |
|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.0100 | 0.0116 | 0.0079 | 0.0092 | 0.0107 |
| 0.98 | 0.0205 | 0.0228 | 0.0161 | 0.0182 | 0.0208 |
| 0.95 | 0.0519 | 0.0548 | 0.0409 | 0.0447 | 0.0499 |
| 0.9 | 0.1040 | 0.1063 | 0.0834 | 0.0887 | 0.0965 |
| 0.85 | 0.1544 | 0.1564 | 0.1267 | 0.1324 | 0.1417 |
| 0.8 | 0.2036 | 0.2061 | 0.1711 | 0.1768 | 0.1870 |

## Adjacent-Layer Attention Similarity

| layer pair | global cosine | global JSD | point-query cosine | point-query JSD |
|---|---:|---:|---:|---:|
| 0-1 | 0.1750 | 0.1033 | 0.2219 | 0.2840 |
| 1-2 | 0.9471 | 0.0126 | 0.4305 | 0.1830 |
| 2-3 | 0.9604 | 0.0093 | 0.4964 | 0.1398 |
| 3-4 | 0.9716 | 0.0070 | 0.5600 | 0.1353 |
| 4-5 | 0.9587 | 0.0106 | 0.6343 | 0.1241 |
| 5-6 | 0.9414 | 0.0152 | 0.7696 | 0.0982 |
| 6-7 | 0.9442 | 0.0146 | 0.7782 | 0.0888 |
| 7-8 | 0.9500 | 0.0128 | 0.8529 | 0.0686 |
| 8-9 | 0.9379 | 0.0157 | 0.8219 | 0.0813 |
| 9-10 | 0.9443 | 0.0145 | 0.8218 | 0.0749 |
| 10-11 | 0.9369 | 0.0163 | 0.8127 | 0.0659 |

## Top Regression Examples

Sorted by readout-layer deleted attention mass.

| keep | image | pair | scene | dense margin | pruned margin | endpoint deleted | top nonself deleted | readout deleted attention |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 0.8 | images/non_real/52759065465_1704ece345_k.jpg | 1 | non_real | 0.0908 | -0.0763 | 1.0000 | 1.0000 | 0.6635 |
| 0.85 | images/non_real/52759065465_1704ece345_k.jpg | 1 | non_real | 0.0908 | -0.1026 | 1.0000 | 1.0000 | 0.6474 |
| 0.8 | images/outdoor/51559847349_dad8fdba89_k.jpg | 0 | outdoor | 1.3085 | -1.0669 | 1.0000 | 1.0000 | 0.6370 |
| 0.8 | images/outdoor/53341440575_1425c84d51_k.jpg | 1 | outdoor | 1.2936 | -0.7948 | 1.0000 | 1.0000 | 0.6349 |
| 0.85 | images/outdoor/53341440575_1425c84d51_k.jpg | 1 | outdoor | 1.2936 | -0.8154 | 1.0000 | 1.0000 | 0.6249 |
| 0.8 | images/indoor/25983921221_d39847a674_k.jpg | 1 | indoor | 0.5626 | -0.1950 | 1.0000 | 1.0000 | 0.6154 |
| 0.8 | images/adverse_style/51658483222_f5e2a89be6_k.jpg | 1 | adverse_style | 0.1373 | -0.0612 | 1.0000 | 1.0000 | 0.6122 |
| 0.9 | images/outdoor/53341440575_1425c84d51_k.jpg | 1 | outdoor | 1.2936 | -0.6762 | 1.0000 | 1.0000 | 0.6091 |
| 0.9 | images/non_real/52759065465_1704ece345_k.jpg | 1 | non_real | 0.0908 | -0.0370 | 1.0000 | 1.0000 | 0.5641 |
| 0.85 | images/outdoor/51559847349_dad8fdba89_k.jpg | 0 | outdoor | 1.3085 | -1.0470 | 1.0000 | 1.0000 | 0.5606 |
| 0.8 | images/non_real/53692302701_2bd501a7fb_b.jpg | 1 | non_real | 3.0143 | -0.6427 | 1.0000 | 1.0000 | 0.5583 |
| 0.8 | images/adverse_style/10565642806_6362d17cf0_k.jpg | 1 | adverse_style | 0.7428 | -1.0794 | 1.0000 | 1.0000 | 0.5465 |
| 0.8 | images/indoor/30497591893_e5f7917957_o.jpg | 0 | indoor | 0.4989 | -0.4568 | 1.0000 | 1.0000 | 0.5464 |
| 0.8 | images/adverse_style/51821788430_7b83368c03_k.jpg | 0 | adverse_style | 0.7325 | -1.6238 | 1.0000 | 1.0000 | 0.5284 |
| 0.8 | images/underwater/42189634014_b83b0a21a9_k.jpg | 1 | underwater | 1.2847 | -0.9079 | 1.0000 | 1.0000 | 0.5279 |
| 0.8 | images/adverse_style/45752608265_7992a16765_k.jpg | 1 | adverse_style | 0.0360 | -0.0432 | 1.0000 | 1.0000 | 0.5122 |
| 0.8 | images/outdoor/22884041219_3e5cdc64c4_k.jpg | 0 | outdoor | 0.0948 | -0.3714 | 1.0000 | 1.0000 | 0.4910 |
| 0.8 | images/underwater/42189634014_b83b0a21a9_k.jpg | 0 | underwater | 0.7293 | -1.8522 | 1.0000 | 1.0000 | 0.4838 |
| 0.85 | images/non_real/53692302701_2bd501a7fb_b.jpg | 1 | non_real | 3.0143 | -0.6204 | 1.0000 | 1.0000 | 0.4772 |
| 0.85 | images/adverse_style/51821788430_7b83368c03_k.jpg | 0 | adverse_style | 0.7325 | -1.3602 | 1.0000 | 1.0000 | 0.4747 |
| 0.8 | images/underwater/33160416385_2318c69954_h.jpg | 1 | underwater | 1.1738 | -2.2587 | 1.0000 | 1.0000 | 0.4687 |
| 0.8 | images/non_real/52231534579_1b59241822_k.jpg | 1 | non_real | 0.9456 | -0.1824 | 1.0000 | 1.0000 | 0.4670 |
| 0.8 | images/object/123949917_fd08c80d60_b.jpg | 0 | object | 1.0854 | -1.7687 | 1.0000 | 1.0000 | 0.4627 |
| 0.8 | images/object/19810710865_42df884006_h.jpg | 0 | object | 0.2848 | -0.0312 | 1.0000 | 1.0000 | 0.4458 |
| 0.8 | images/underwater/23433401544_111a517ca0_k.jpg | 1 | underwater | 0.1858 | -0.5516 | 1.0000 | 1.0000 | 0.4392 |
