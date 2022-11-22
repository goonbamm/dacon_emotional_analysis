# 월간 데이콘 발화자의 감정인식 AI 경진대회

<br>

## Performance 📊
----

<br>

|model_name| dev macro F1 score | test macro F1 score |
|:--:|:--:|:--:|
| Baseline | 0.4507 | 0.3890 |
| emoBERTa-Base | 0.5901 | 0.4752 |
| emoBERTa-Base, accumulate = 3 | 0.5901 | 0.4705 |
| **emoBERTa-Large** | **0.6627** | **0.5153** |
| CoMPM: roBERTa-Large (default) | 0.5381 | 0.3818 |
| CoMPM: emoBERTa-Large (freeze) | 0.6282 | - |
| CoMPM: emoBERTa-Large (no freeze) | 0.6342 | - |
| CoMPM: emoBERTa-Large (freeze, focal_loss) | 0.6405 | 0.3845 |

<br>

* acculmulate 는 이전 발화를 함께 데이터로 넣고자, 가볍게 전처리를 해주었습니다. 현재 발화까지 포함해서 최대 3개까지 갖고 있도록 합니다. 이전 발화가 없으면, 없는 대로 둡니다.

<br>

## To Do 📚
----

<br>

- COMET 데이터 활용

    + emoBERTa 기반으로 활용할 예정

<br>

- emoBERTa

    + emoBERTa 논문 정독

<br>

- Graph 구조를 활용해볼 순 없을까?

<br>

- Label Smoothing?

<br>

## Done ⭐
----

<br>

- Baseline

    + 코드 간결화

    + EarlyStop 적용

    + AdamW 변경

    + AMP 적용

<br>

- 데이터 분석하기

    + train / valid 비율

    + 대화 개수 / 담화 개수 측정

<br>

- [CoMPM](https://github.com/rungjoo/CoMPM)

    + [CoMPM 논문 정독](https://heygeronimo.tistory.com/32)

    + 코드 수정 및 DACON DATASET 적용

    + 성능 측정

        * whether freeze or not

        * pretrained model: [RoBERTa-Large](https://huggingface.co/roberta-large) / [Emoberta-Large](https://huggingface.co/tae898/emoberta-large?text=I+like+you.+I+love+you)

        * loss: CrossEntropyLoss or Focal Loss



<br>

- [COMET-ATOMIC-2020](https://github.com/allenai/comet-atomic-2020)

    + 모델로 의도, 장소, 원인 등 다양한 정보 추출

    + 총 51개의 relation 을 뽑을 것이며, 1개당 1시간이 소요되기 때문에 50시간이 지나야 끝날 것 같다.
