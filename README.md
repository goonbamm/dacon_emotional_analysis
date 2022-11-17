# 월간 데이콘 발화자의 감정인식 AI 경진대회

<br>

## To Do 📚
----

<br>

- 데이터 분석하기
    + train / valid 비율
    + 대화 개수 / 담화 개수 측정

<br>

- CoMPM 을 Dacon Dataset 으로 Fine Tuning 해보기
    + 현재 데이터 형식은 EmoryNLP 로 변환했음.

<br>

## Done ⭐
----

<br>

- 코드 간결화

- EarlyStop 적용

- AdamW 변경

- AMP 적용

<br>


- [CoMPM 논문 정독](https://heygeronimo.tistory.com/32)

    + GPU 소모량이 커서 AMP 적용했으나, 여전히 Batch_Size = 1 인데도 돌아가지 않음.

    + Large 가 아닌 Base 에서 성능 확인

        * 69.46 에서 58.04 로 떨어짐.

        * 확실히 Large 모델을 사용할 수 있으면 좋으나, 14GB 이상 GPU 가 필요하다.

        * checkpoint 가 14GB 라서 일단 받아둠.

        * Base 로 시도 다 해보고, 마지막에 Large 를 써봐야겠다.

<br>

- [COMET-ATOMIC-2020](https://github.com/allenai/comet-atomic-2020) 모델로 의도, 장소 등 다양한 정보 추출

    + 총 51개의 relation 을 뽑을 것이며, 1개당 1시간이 소요되기 때문에 50시간이 지나야 끝날 것 같다.