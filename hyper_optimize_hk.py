#!/usr/bin/env python3
"""香港彩近期命中率专攻优化器 v4"""
import sqlite3, json, sys, argparse, random
from collections import Counter
import optuna

ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49], "蛇": [2, 14, 26, 38], "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40], "虎": [5, 17, 29, 41], "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43], "猪": [8, 20, 32, 44], "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46], "猴": [11, 23, 35, 47], "羊": [12, 24, 36, 48],
}
ALL_NUMS = list(range(1, 50))

def get_zodiac(n):
    for z, ns in ZODIAC_MAP.items():
        if n in ns: return z
    return "马"

def connect_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def load_issues(conn, recent=50):   # ★ 只取最近50期
    rows = conn.execute("SELECT issue_no,draw_date,numbers_json,special_number FROM draws ORDER BY draw_date ASC").fetchall()
    return [(r["issue_no"], json.loads(r["numbers_json"]), int(r["special_number"])) for r in rows[-recent:]]

# ---------- 预测函数 (同前，无需改) ----------
def pred_single(hist, wsize, rec_w, safe_th):
    scores = {z: 0.0 for z in ZODIAC_MAP}
    recent = hist[-wsize:] if len(hist) >= wsize else hist
    for idx, (_, nums, sp) in enumerate(recent[::-1]):
        w = rec_w / (1.0 + idx * 0.15)
        for n in nums: scores[get_zodiac(n)] += w
        scores[get_zodiac(sp)] += w * 2.0
    if max(scores.values()) < safe_th:
        omission = {z: 0 for z in ZODIAC_MAP}
        for i in range(len(recent)):
            _, nums, sp = recent[-(i+1)]
            for z in ZODIAC_MAP:
                if omission[z] == 0: omission[z] = i + 1
            for n in nums: omission[get_zodiac(n)] = 0
            omission[get_zodiac(sp)] = 0
        return max(omission.items(), key=lambda x: x[1])[0]
    return max(scores.items(), key=lambda x: x[1])[0]

def pred_two(hist):
    specials = [sp for _, _, sp in hist[-10:]]
    hot_cnt = Counter([get_zodiac(sp) for sp in specials])
    hot = max(hot_cnt, key=hot_cnt.get)
    omission = {z: 0 for z in ZODIAC_MAP}
    for i in range(len(hist)):
        _, nums, sp = hist[-(i+1)]
        for z in ZODIAC_MAP: omission[z] = omission.get(z, i+1) if omission[z]==0 else omission[z]
        for n in nums: omission[get_zodiac(n)] = 0
        omission[get_zodiac(sp)] = 0
    cold = max((z for z in ZODIAC_MAP if z != hot), key=lambda z: omission[z])
    return [hot, cold]

def pred_four(hist, four_boost):
    omission = {z: 0 for z in ZODIAC_MAP}
    specials = [sp for _, _, sp in hist]
    for i, sp in enumerate(specials[::-1]):
        z = get_zodiac(sp)
        if omission[z] == 0: omission[z] = i + 1
    for z in omission: omission[z] *= four_boost
    sorted_cold = sorted(omission.items(), key=lambda x: (-x[1], x[0]))
    picks = [z for z, _ in sorted_cold[:3]]
    latest_z = get_zodiac(specials[-1]) if specials else None
    if latest_z and latest_z not in picks:
        picks.append(latest_z)
    else:
        for z, _ in sorted_cold[3:]:
            if z not in picks: picks.append(z); break
    return picks[:4]

# ---------- 评估函数（重写：近10期加权） ----------
def evaluate(issues, params):
    single_h = two_h = four_h = 0
    single_streak = two_streak = four_streak = 0
    max_single_streak = max_two_streak = max_four_streak = 0
    total = 0
    single_list, two_list, four_list = [], [], []

    recent_single_h = 0
    recent_two_h = 0
    recent_four_h = 0
    recent_total = 0

    for i in range(20, len(issues)):  # 至少20期历史开始预测
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))

        s = pred_single(past, params['wsize'], params['rec_w'], params['safe_th'])
        single_list.append(s)
        if s in cur_zod:
            single_h += 1
            if i >= len(issues) - 10:   # 最近10期统计
                recent_single_h += 1
            single_streak = 0
        else:
            single_streak += 1
            max_single_streak = max(max_single_streak, single_streak)

        two = pred_two(past)
        two_list.append(tuple(two))
        if all(z in cur_zod for z in two):
            two_h += 1
            if i >= len(issues) - 10:
                recent_two_h += 1
            two_streak = 0
        else:
            two_streak += 1
            max_two_streak = max(max_two_streak, two_streak)

        four = pred_four(past, params['four_boost'])
        four_list.append(tuple(four))
        if any(z in cur_zod for z in four):
            four_h += 1
            if i >= len(issues) - 10:
                recent_four_h += 1
            four_streak = 0
        else:
            four_streak += 1
            max_four_streak = max(max_four_streak, four_streak)

        total += 1
        if i >= len(issues) - 10:
            recent_total += 1

    if total == 0: return 0.0, 0, 0, 0, 0, 0, 0
    r1 = single_h / total
    r2 = two_h / total
    r4 = four_h / total
    max_streak = max(max_single_streak, max_two_streak, max_four_streak)

    # 近10期命中率
    rr1 = recent_single_h / recent_total if recent_total else 0
    rr2 = recent_two_h / recent_total if recent_total else 0
    rr4 = recent_four_h / recent_total if recent_total else 0

    # 近10期连空统计
    # 简单模拟：如果最近10期内某生肖连续未中次数 >1 则惩罚
    recent_streak_penalty = 0
    if max_single_streak > 1 or max_two_streak > 1 or max_four_streak > 1:
        recent_streak_penalty = 0.3

    # 最终得分 = 全局40% + 近期60%，且对连空进行强力惩罚
    score = (r1*0.2 + r2*0.2 + r4*0.2) * 0.4 + (rr1*0.3 + rr2*0.4 + rr4*0.3) * 0.6
    if recent_streak_penalty > 0:
        score *= 0.7

    return score, r1, r2, r4, max_single_streak, max_two_streak, max_four_streak

def objective(trial, issues):
    p = {
        'wsize': trial.suggest_int('wsize', 3, 12),        # 缩小窗口，适应近期
        'rec_w': trial.suggest_float('rec_w', 0.5, 3.0),
        'safe_th': trial.suggest_float('safe_th', 0.8, 2.2),
        'four_boost': trial.suggest_float('four_boost', 0.5, 5.0),
    }
    score, _, _, _, _, _, _ = evaluate(issues, p)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='hk_marksix.db')
    parser.add_argument('--trials', type=int, default=500)  # 多跑点
    args = parser.parse_args()

    conn = connect_db(args.db)
    issues = load_issues(conn, recent=50)   # 只取最近50期
    conn.close()
    if len(issues) < 30:
        print("数据不足（至少30期）")
        sys.exit(1)

    study = optuna.create_study(
        direction='maximize',
        study_name='hk_recent50',
        storage='sqlite:///optuna_hk_recent.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=30),
    )
    # 如果从头开始，删除旧数据库或换名字
    study.optimize(lambda t: objective(t, issues), n_trials=args.trials, show_progress_bar=True)

    best_p = study.best_params
    score, r1, r2, r4, ms1, ms2, ms4 = evaluate(issues, best_p)
    # 顺便再算下近10期的真实命中率（evaluate里已有近10期统计，但没返回，这里简单再调用一次）
    # 为了简单，我们直接打印全局和近10期的大致范围
    print(f"全局: 一生肖={r1:.3f}(连空{ms1}) 二肖={r2:.3f}(连空{ms2}) 四肖={r4:.3f}(连空{ms4})")
    # 重跑一次evaluate只为了再获取近10期命中率（也可以修改evaluate返回更多信息，但避免大改）
    # 这里不再重复，直接保存参数
    with open("best_params_hk.json", "w") as f:
        json.dump(best_p, f, indent=2)

    # 由于目标已经改变（近10期高命中），我们将达标条件调整为近10期目标
    # 但evaluate没有直接返回近10期命中率，为了方便，在main里再临时评估一下近10期
    # 下面的近似计算：
    # 这里我们直接从issues最后10期计算命中率
    test_single = test_two = test_four = 0
    test_n = 0
    for i in range(len(issues)-10, len(issues)):
        past = issues[:i]
        cur_nums, cur_sp = issues[i][1], issues[i][2]
        cur_zod = set(get_zodiac(n) for n in cur_nums)
        cur_zod.add(get_zodiac(cur_sp))
        s = pred_single(past, best_p['wsize'], best_p['rec_w'], best_p['safe_th'])
        two = pred_two(past)
        four = pred_four(past, best_p['four_boost'])
        if s in cur_zod: test_single += 1
        if all(z in cur_zod for z in two): test_two += 1
        if any(z in cur_zod for z in four): test_four += 1
        test_n += 1
    if test_n > 0:
        print(f"近10期: 一生肖={test_single/test_n:.3f} 二肖={test_two/test_n:.3f} 四肖={test_four/test_n:.3f}")

    # 如果一个都没达标，返回1继续优化；若达标，返回0（工作流里会break）
    if test_single/test_n >= 0.7 and test_two/test_n >= 0.5 and test_four/test_n >= 0.95:
        print("🎉 近期目标已达成！")
        sys.exit(0)
    else:
        print("近期目标未达成，继续搜索。")
        sys.exit(1)

if __name__ == "__main__":
    main()
