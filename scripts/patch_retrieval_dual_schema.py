"""Patch retrieval_server.py QueryRequest to accept both formats.

SkyRL canonical: {"queries": [str], "topk": int, ...}
SkyRL skyrl-gym call_search_api: {"query": str, "topk": int, ...}

This patch makes both work.
"""
import re
import sys

f = sys.argv[1]
src = open(f).read()

new_request = '''class QueryRequest(BaseModel):
    """patch (agenicRL): accept both formats:
       - {"queries": [str], "topk": int, "return_scores": bool}
       - {"query": str, "topk": int, "return_scores": bool}
    """
    queries: Optional[List[str]] = None
    query: Optional[str] = None
    topk: Optional[int] = None
    return_scores: bool = False

    def get_queries(self) -> List[str]:
        if self.queries:
            return self.queries
        if self.query:
            return [self.query]
        return []
'''

src = re.sub(
    r"class QueryRequest\(BaseModel\):.*?return_scores: bool = False\n",
    new_request,
    src,
    count=1,
    flags=re.DOTALL,
)

# All callers should use get_queries()
src = src.replace("request.queries", "request.get_queries()")

open(f, "w").write(src)
print("patched OK")
