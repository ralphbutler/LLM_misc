import asyncio

from duckduckgo_search import AsyncDDGS

async def aget_results(word):
    results = await AsyncDDGS(proxy=None).atext(word, max_results=100)
    return results

async def main():
    query = "what is the weather in Madrid tomorrow?"
    results = await AsyncDDGS().atext(query, region='wt-wt', safesearch='off',
                                      timelimit='y', max_results=1)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
