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
| emoBERTa-Large | 0.6627 | 0.5153 |
| CoMPM: roBERTa-Large (default) | 0.5381 | 0.3818 |
| CoMPM: emoBERTa-Large (freeze) | 0.6282 | - |
| CoMPM: emoBERTa-Large (no freeze) | 0.6342 | - |
| CoMPM: emoBERTa-Large (freeze, focal_loss) | 0.6405 | 0.3845 |
| **emoBERTa-Large: COMET - isFilledBy** | **0.6676** | **0.5226** |

<br>

* acculmulate 는 이전 발화를 함께 데이터로 넣고자, 가볍게 전처리를 해주었습니다. 현재 발화까지 포함해서 최대 3개까지 갖고 있도록 합니다. 이전 발화가 없으면, 없는 대로 둡니다.

<br>

## To Do 📚
----

<br>

- [EmoBERTa](https://github.com/tae898/erc)

    + 코드 적용 및 실험 제출

<br>

- Test Valid 와의 Score 의 격차가 심한데, 혹시 overfitting, underfitting 이 일어나고 있는 것은 아닐까?

    + epoch 별로 제출해보고 있음.

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

    + 51개 항목 모두 생성 완료

    + None 비율이 30% 가 넘지 않는 데이터만 실험에 사용

    + 실험 사용 완료 (결과는 COMET 폴더 참고)

<br>

- [EmoBERTa](https://github.com/tae898/erc)

    + [CoMPM 논문 정독](https://heygeronimo.tistory.com/33)