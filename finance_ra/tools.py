from datetime import date
from typing import Optional, Type

from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

import yfinance as yf


class TickerInput(BaseModel):
    ticker: str = Field(description="A unique identifier or symbol assigned to a publicly traded company or financial instrument")

class CompanyInfoRetriever(BaseTool):
    name: str = "company_information_retriever"
    description: str = "This tool retrieves key company information, including its address, industry, sector, company executives, business summary, and official website. It also provides financial details such as market capitalization, current stock price, EBITDA, total debt, total revenue, and debt-to-equity ratio."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        company = yf.Ticker(ticker)
        company_info = company.get_info()
        return company_info

class DividendEarningsDateRetriever(BaseTool):
    name: str = "last_dividend_earnings_date_retriever"
    description: str = "This tool retrieves a company's last dividend date and earnings release dates. It does not provide information about historical dividend yields."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.get_calendar()

class MutualFundHoldersRetriever(BaseTool):
    name: str = "mutual_fund_holders_retriever"
    description: str = "This tool retrieves a company's top mutual fund holders, including their percentage of shares, stock count, and value of holdings."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        ticker_obj = yf.Ticker(ticker)
        mf_holders = ticker_obj.get_mutualfund_holders()
        return mf_holders.to_dict(orient="records")

class InstitutionalHoldersRetriever(BaseTool):
    name: str = "institutional_holders_retriever"
    description: str = "This tool retrieves a company's top institutional holders, including their percentage of shares, stock count, and value of holdings."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        ticker_obj = yf.Ticker(ticker)
        inst_holders = ticker_obj.get_institutional_holders()
        return inst_holders.to_dict(orient="records")

class StockUpgradesDowngradesRetriever(BaseTool):
    name: str = "stock_grade_upgrades_downgrades_retriever"
    description: str = "This tool retrieves stock rating upgrades and downgrades, providing details such as firm names, 'To Grade' and 'From Grade' changes, and the date of the rating action."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        ticker_obj = yf.Ticker(ticker)
        curr_year = date.today().year
        upgrades_downgrades = ticker_obj.get_upgrades_downgrades()
        upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index > f"{curr_year}-01-01"]
        upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["Action"].isin(["up", "down"])]
        return upgrades_downgrades.to_dict(orient="records")

class StockSplitsHistoryRetriever(BaseTool):
    name: str = "stock_splits_history_retriever"
    description: str = "This tool retrieves a company's historical stock splits data."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        ticker_obj = yf.Ticker(ticker)
        hist_splits = ticker_obj.get_splits()
        return hist_splits.to_dict()

class StockNewsRetriever(BaseTool):
    name: str = "stock_news_retriever"
    description: str = "This tool retrieves the latest news articles discussing a particular stock ticker."
    args_schema: Type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> dict:
        """Use the Tool"""
        
        num_articles = 5
        ticker_obj = yf.Ticker(ticker)
        all_articles = ticker_obj.news
        company_news = ""
        for index, article in enumerate(all_articles[:num_articles]):
            company_news += f"\nNEWS {index+1}\n{'--'*50}\nURL: {article['content'].get('clickThroughUrl')}\nSummary: {article['content']['summary']}\n"
            company_news += "=="*50
        return company_news
