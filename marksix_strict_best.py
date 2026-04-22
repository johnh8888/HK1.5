import json
from collections import Counter
from pathlib import Path

from marksix_local import (
    connect_db,
    init_db,
    _draws_ordered_asc,
    _weighted_consensus_pools,
    _trio3_generators,
    _special_generators,
    get_strategy_weights,
    get_strategy_health,
    get_single_zodiac_pick,
    get_two_zodiac_picks,
    get_three_zodiac_picks,
    get_texiao4_picks,
    get_top_special_votes,
    get_special_recommendation,
    get_zodiac_by_number,
    get_latest_draw,
    print_final_recommendation,
    print_dashboard,
)


def best_strategy_snapshot(db_path: str):
    conn = connect_db(db_path)
    try:
        init_db(conn)
        latest = get_latest_draw(conn)
        if not latest:
            print("暂无开奖数据")
            return
        issue_no = latest["issue_no"]
        main6, p10, p14, pool20, special = _weighted_consensus_pools(conn, issue_no, status="REVIEWED")
        print(f"最新期号: {issue_no}")
        print(f"共识6码: {main6}")
        print(f"共识10码: {p10}")
        print(f"共识14码: {p14}")
        print(f"共识20码: {pool20}")
        print(f"共识特号: {special}")

        trio_gens = _trio3_generators(conn, issue_no, status="REVIEWED")
        print("\n三中三候选:")
        for name, nums in trio_gens.items():
            print(f"  {name}: {nums}")

        if main6 and pool20:
            special_gens = _special_generators(conn, issue_no, main6, pool20, status="REVIEWED")
            print("\n特号候选:")
            for name, nums in special_gens.items():
                print(f"  {name}: {nums[:8]}")

        print("\n生肖推荐:")
        print("  单生肖:", get_single_zodiac_pick(conn, issue_no))
        print("  双生肖:", get_two_zodiac_picks(conn, issue_no))
        print("  三生肖:", get_three_zodiac_picks(conn, issue_no))
        print("  特肖4只:", get_texiao4_picks(conn, issue_no))

        print("\n策略健康度:")
        health = get_strategy_health(conn, window=10)
        for k, v in health.items():
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")

        print("\n当前最强推荐:")
        print_final_recommendation(conn)
        print("\n仪表盘:")
        print_dashboard(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    best_strategy_snapshot(str(Path(__file__).with_name("hk_marksix.db")))
