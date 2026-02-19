conda create -n py310 python=3.10
conda activate py310
conda deactivate 

pip install -r requirements.txt

sqlite3 ecommerce.db


Create a new table that summarizes weekly orders by customer state for the year 2017.
Before deciding the final SQL flow, generate at least three alternative strategies,
execute each one, compare the results, and select the best strategy.
Explain why the selected flow is optimal.

Create a new table that summarizes weekly orders by customer state for 2017.
Generate three alternative SQL strategies and select the best.

Who are the top 10 sellers?

What are the most popular payment methods?


지난 3개월 주문 데이터를 기준으로,

전체 고객에게 5% 할인 쿠폰,

전체 고객에게 10% 할인 쿠폰,

최근 1년 동안 결제금액 상위 20% 고객에게만 15% 쿠폰

2017년 10월 1일부터 2017년 12월 31일까지의 주문 데이터를 기준으로,

1. 전체 고객에게 5% 할인 쿠폰,
2. 전체 고객에게 10% 할인 쿠폰,
3. 2017년 결제액 상위 20% 고객에게만 15% 쿠폰

이 세 가지 전략을 각각 브랜치로 만들어서 매출과 주문 건수를 비교해줘.
그리고 세 전략 중에서 가장 좋은 전략을 선택해서, 그 브랜치를 메인 DB에 자동으로 커밋해줘.




2017년 10월 1일부터 2017년 12월 31일까지의 주문 데이터를 기준으로,

1. 전체 고객에게 1% 할인 쿠폰,
2. 전체 고객에게 5% 할인 쿠폰,
3. 전체 고객에게 10% 할인 쿠폰,

이 세 가지 전략을 각각 브랜치로 만들어서 매출과 주문 건수를 비교해줘.
그리고 세 전략 중에서 가장 좋은 전략을 선택해서, 그 브랜치를 메인 DB에 자동으로 커밋해줘.

2017년 10월 1일부터 2017년 12월 31일까지의 주문 데이터를 기준으로,

1. 전체 주문 중 장바구니 금액이 100 이상인 주문에만 5% 할인 쿠폰,
2. 장바구니 금액이 200 이상인 주문에만 10% 할인 쿠폰,
3. 장바구니 금액이 300 이상인 주문에만 15% 할인 쿠폰,

이 세 가지 전략을 각각 브랜치로 만들어서 매출(total_revenue)과 주문 건수(order_count)를 비교해줘.
그리고 세 전략 중에서 가장 좋은 전략을 선택해서, 그 브랜치를 메인 DB에 자동으로 커밋해줘.



1. 2017년 결제액 상위 20% 고객에 대해서만 15% 쿠폰




2017년 10월 1일부터 2017년 12월 31일까지의 전체 주문 데이터를 기준으로,
각 주문에 대해 장바구니 금액(상품 가격 + 배송비)을 계산한 뒤 다음 세 가지 전략을 비교해줘.

전략 1: 장바구니 금액이 100 이상인 주문에만 5% 할인을 적용하고, 100 미만 주문은 할인을 적용하지 않는다.

전략 2: 장바구니 금액이 200 이상인 주문에만 10% 할인을 적용하고, 200 미만 주문은 할인을 적용하지 않는다.

전략 3: 장바구니 금액이 300 이상인 주문에만 15% 할인을 적용하고, 300 미만 주문은 할인을 적용하지 않는다.

각 전략에 대해, 할인 대상이 아닌 주문도 포함한 전체 주문 집합을 기준으로 할인 적용 후 총 매출(total_revenue)과 전체 주문 건수(order_count)를 계산해줘.
세 전략을 각각 시뮬레이션한 뒤, 최종적으로 어떤 전략이 전체 매출과 주문 수 측면에서 가장 좋은지 평가해줘