import urllib.request
import urllib.parse
import json
import re
import html

class WebSearch:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }

    def clean_query(self, raw_query):
        """Strips conversational boilerplate to isolate the core search keywords."""
        q = raw_query.strip().rstrip('?.!')
        
        # Remove trailing search instructions like 'you can search on internet', 'search on web', etc.
        patterns_to_remove = [
            r'(\s+)?\b(you can\s+)?(search|look up|find|check)(\s+it)?(\s+on\s+(the\s+)?(internet|web|google))?\b',
            r'\b(on\s+(the\s+)?(internet|web|google))\b',
            r'^tell me (about|the)?\s*',
            r'^can you (please\s*)?(search|find|tell me|look up)\s*(about|for)?\s*',
            r'^please (search|find|tell me|look up)\s*(about|for)?\s*',
            r'^(what is|what are|who is|who was|where is|how much is)\s*(the\s*)?',
            r'^(search|look up|find|give me)\s*(for|about|the)?\s*'
        ]
        
        for p in patterns_to_remove:
            q = re.sub(p, ' ', q, flags=re.IGNORECASE).strip()

        # Re-compact spaces
        q = ' '.join(q.split())
        return q if len(q) > 2 else raw_query.strip().rstrip('?.!')

    def search(self, query, location="India"):
        """Performs multi-source web search with optional regional localization."""
        cleaned = self.clean_query(query)
        
        # Check if query already specifies a country/region
        has_country = any(c in cleaned.lower() for c in [
            "india", "usa", "us", "uk", "canada", "dubai", "uae", "pakistan", 
            "australia", "germany", "japan", "in inr", "in usd", "in rupees", "in dollars"
        ])
        
        # If searching for prices/stores without a specified country, localize to user's region
        is_price_query = any(k in cleaned.lower() for k in ["price", "cost", "how much", "rate", "buy", "store"])
        if is_price_query and not has_country and location:
            search_query = f"{cleaned} in {location} price"
        else:
            search_query = cleaned

        print(f"[THINKING] Cleaned search query: '{search_query}' (from '{query}')")
        
        # 1. Try DuckDuckGo Lite / HTML Search with region targeting
        region_code = "in-en" if (location and location.lower() == "india") else ""
        ddg_results = self._search_duckduckgo(search_query, region_code=region_code)
        if ddg_results:
            return ddg_results

        # 2. Try DuckDuckGo Instant Answer API
        ddg_api = self._search_ddg_api(search_query)
        if ddg_api:
            return ddg_api

        # 3. Try Wikipedia Search & Summary
        wiki_res = self._search_wikipedia(cleaned)
        if wiki_res:
            return wiki_res

        return None

    def _search_duckduckgo(self, query, region_code="in-en"):
        try:
            # DuckDuckGo Lite endpoint with region parameter
            params = {'q': query}
            if region_code:
                params['kl'] = region_code
            data = urllib.parse.urlencode(params).encode('utf-8')
            req = urllib.request.Request(
                'https://lite.duckduckgo.com/lite/',
                data=data,
                headers=self.headers
            )
            with urllib.request.urlopen(req, timeout=6) as res:
                content = res.read().decode('utf-8', errors='ignore')

            # Extract result snippets from DDG Lite table
            snippets = re.findall(r'class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', content, re.DOTALL | re.IGNORECASE)
            if not snippets:
                snippets = re.findall(r'<td[^>]*class=[\'"][^\'"]*snippet[^\'"]*[\'"][^>]*>(.*?)</td>', content, re.DOTALL | re.IGNORECASE)

            cleaned_snippets = []
            for s in snippets[:3]:
                text = re.sub(r'<[^>]+>', '', s)
                text = html.unescape(text).strip()
                text = ' '.join(text.split())
                if len(text) > 20 and text not in cleaned_snippets:
                    cleaned_snippets.append(text)

            if cleaned_snippets:
                return " ".join(cleaned_snippets)
        except Exception as e:
            print(f"[SEARCH] DuckDuckGo Lite query error: {e}")

        # Fallback to HTML endpoint
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=6) as res:
                content = res.read().decode('utf-8', errors='ignore')

            snippets = re.findall(r'class=[\'"]result__snippet[^\'"]*[\'"][^>]*>(.*?)</a>', content, re.DOTALL | re.IGNORECASE)
            cleaned_snippets = []
            for s in snippets[:3]:
                text = re.sub(r'<[^>]+>', '', s)
                text = html.unescape(text).strip()
                text = ' '.join(text.split())
                if len(text) > 20 and text not in cleaned_snippets:
                    cleaned_snippets.append(text)

            if cleaned_snippets:
                return " ".join(cleaned_snippets)
        except Exception as e:
            print(f"[SEARCH] DuckDuckGo HTML query error: {e}")

        return None

    def _search_ddg_api(self, query):
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=5) as res:
                data = json.loads(res.read().decode('utf-8'))
                if data.get('AbstractText'):
                    return data['AbstractText']
                elif data.get('Answer'):
                    return data['Answer']
        except Exception:
            pass
        return None

    def _search_wikipedia(self, query):
        try:
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'MonicaAI/1.0'})
            with urllib.request.urlopen(req, timeout=5) as res:
                data = json.loads(res.read().decode('utf-8'))
                results = data.get('query', {}).get('search', [])
                if results:
                    best_title = results[0]['title']
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(best_title)}"
                    req2 = urllib.request.Request(summary_url, headers={'User-Agent': 'MonicaAI/1.0'})
                    with urllib.request.urlopen(req2, timeout=5) as res2:
                        summary_data = json.loads(res2.read().decode('utf-8'))
                        if summary_data.get('extract'):
                            sentences = [s.strip() for s in summary_data['extract'].split('. ') if s.strip()]
                            return '. '.join(sentences[:2]).strip() + '.'
        except Exception:
            pass
        return None
