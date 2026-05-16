from fastapi import HTTPException
from app.db.tracker_repo import TrackerRepo
from app.utils.tracker_utils import (
    get_prior_quarter,
    calculate_qoq_change,
    calculate_trend_direction,
)
from app.schemas.tracker import (
    FundListResponse,
    FundListRow,
    FundHoldingsResponse,
    HoldingRow,
    FundOverlapResponse,
    OverlapRow,
    HoldingHistoryResponse,
    HistoryRow,
)


class TrackerService:
    def __init__(self, repo: TrackerRepo):
        self.repo = repo

    def get_funds(self, sort: str, search: str) -> FundListResponse:
        rows = self.repo.get_fund_list()
        funds = []
        for cik, name, quarter, active, count, value in rows:
            if search and search.lower() not in (name or "").lower():
                continue
            funds.append(
                FundListRow(
                    cik=cik,
                    name=name,
                    latest_quarter=quarter,
                    is_active=active,
                    holding_count=count,
                    total_value_usd=value,
                )
            )

        if sort == "holdings":
            funds.sort(key=lambda f: f.holding_count, reverse=True)
        elif sort == "name":
            funds.sort(key=lambda f: (f.name or "").lower())
        else:
            funds.sort(key=lambda f: f.total_value_usd, reverse=True)

        return FundListResponse(funds=funds, count=len(funds))

    def get_fund_holdings(
        self,
        cik: str,
        search: str,
        sort: str,
        order: str,
        page: int,
        limit: int,
        filter_trend: str,
    ) -> FundHoldingsResponse:
        filer = self.repo.get_filer_info(cik)
        if not filer:
            raise HTTPException(status_code=404, detail=f"Filer {cik} not found")

        filer_name, latest_q = filer
        if not latest_q:
            return FundHoldingsResponse(
                cik=cik,
                filer_name=filer_name,
                quarter=None,
                prior_quarter=None,
                total_value_usd=0,
                holdings=[],
                count=0,
                total=0,
                page=page,
                limit=limit,
                message="No filings found — run Backfill to pull EDGAR data",
            )

        prior_q = get_prior_quarter(latest_q)
        current = self.repo.get_holdings_for_quarter(cik, latest_q)

        prior_map = {}
        if prior_q:
            for t, s, v in self.repo.get_holdings_summary_for_quarter(cik, prior_q):
                prior_map[t] = (s, v)

        total_value = sum(r[3] or 0 for r in current) or 1
        holdings = []

        for t, i, s, v, c in current:
            if search and search.lower() not in f"{t} {i}".lower():
                continue
            prev_s, prev_v = prior_map.get(t, (0, 0))
            qoq = calculate_qoq_change(s, prev_s, t not in prior_map)
            holdings.append(
                HoldingRow(
                    ticker=t,
                    name_of_issuer=i,
                    shares=s,
                    value_usd=v,
                    pct_of_portfolio=round(((v or 0) / total_value) * 100, 2),
                    qoq_change=qoq,
                    share_change=(s or 0) - prev_s,
                    cusip=c,
                )
            )

        curr_t = {r[0] for r in current}
        for t, (ps, pv) in prior_map.items():
            if t not in curr_t and (not search or search.lower() in t.lower()):
                holdings.append(
                    HoldingRow(
                        ticker=t,
                        name_of_issuer="",
                        shares=0,
                        value_usd=0,
                        pct_of_portfolio=0,
                        qoq_change="SOLD_OUT",
                        share_change=-ps,
                        cusip="",
                    )
                )

        all_hist = self.repo.get_all_holding_history(cik)
        hist_map = {}
        for t, q, s in all_hist:
            hist_map.setdefault(t, []).append((q, s or 0))

        for h in holdings:
            history = hist_map.get(h.ticker, [])
            if h.qoq_change == "SOLD_OUT" and latest_q:
                if not history or history[-1][0] != latest_q:
                    history = list(history)
                    history.append((latest_q, 0))
            direction, streak, total_change_pct = calculate_trend_direction(history)
            h.trend_direction = direction
            h.trend_streak = streak
            h.total_change_pct = total_change_pct

        if filter_trend != "ALL":
            if filter_trend == "NEW":
                holdings = [h for h in holdings if h.qoq_change == "NEW"]
            elif filter_trend == "SOLD_OUT":
                holdings = [h for h in holdings if h.qoq_change == "SOLD_OUT"]
            elif filter_trend == "TOP_BUYS":
                holdings = [h for h in holdings if h.qoq_change in ("NEW", "ADDED")]
            elif filter_trend == "TOP_SELLS":
                holdings = [
                    h for h in holdings if h.qoq_change in ("SOLD_OUT", "REDUCED")
                ]
            elif filter_trend in ("ACCUMULATING", "DUMPING"):
                holdings = [h for h in holdings if h.trend_direction == filter_trend]
            elif filter_trend in ("ADDED", "REDUCED"):
                holdings = [h for h in holdings if h.qoq_change == filter_trend]

        sort_key = (
            sort
            if sort
            in (
                "ticker",
                "value_usd",
                "shares",
                "pct_of_portfolio",
                "trend_direction",
                "qoq_change",
                "share_change",
                "name_of_issuer",
            )
            else "value_usd"
        )
        holdings.sort(
            key=lambda h: getattr(h, sort_key, 0) or 0, reverse=(order.lower() != "asc")
        )

        total_count = len(holdings)
        offset = (page - 1) * limit
        paginated_holdings = holdings[offset : offset + limit]

        return FundHoldingsResponse(
            cik=cik,
            filer_name=filer_name,
            quarter=latest_q,
            prior_quarter=prior_q,
            total_value_usd=total_value,
            holdings=paginated_holdings,
            count=len(paginated_holdings),
            total=total_count,
            page=page,
            limit=limit,
        )

    def get_overlap(self, min_funds: int) -> FundOverlapResponse:
        rows = self.repo.get_fund_overlap(min_funds)
        overlap = []
        for t, c, v, s, n, issuer in rows:
            overlap.append(
                OverlapRow(
                    ticker=t,
                    name_of_issuer=issuer,
                    fund_count=c,
                    total_value_usd=v,
                    total_shares=s,
                    fund_names=n,
                )
            )
        return FundOverlapResponse(
            overlap=overlap, count=len(overlap), min_funds=min_funds
        )

    def get_holding_history(self, cik: str, ticker: str) -> HoldingHistoryResponse:
        rows = self.repo.get_ticker_history_for_fund(cik, ticker)
        filer = self.repo.get_filer_info(cik)
        filer_name = filer[0] if filer else cik

        history, prev_shares = [], None
        for quarter, filing_date, shares, value in rows:
            history.append(
                HistoryRow(
                    quarter=quarter,
                    filing_date=str(filing_date) if filing_date else None,
                    shares=shares,
                    value_usd=value,
                    share_change=(shares or 0) - prev_shares
                    if prev_shares is not None
                    else None,
                )
            )
            prev_shares = shares or 0

        return HoldingHistoryResponse(
            cik=cik,
            filer_name=filer_name,
            ticker=ticker.upper(),
            history=history,
            quarters_held=len(history),
        )
