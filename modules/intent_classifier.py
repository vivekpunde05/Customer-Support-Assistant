"""
Intent Classification Module
Determines routing decision: RAG answer vs Direct answer vs HITL escalation
"""

import re
from typing import Dict
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import HITL_CONFIDENCE_THRESHOLD, HITL_ESCALATION_KEYWORDS
from modules.llm_client import get_llm, check_setup


class IntentType(str, Enum):
    """Supported intent types for routing decisions."""
    DOCUMENT_QUERY = "document_query"      # Needs RAG retrieval
    GENERAL_CHAT = "general_chat"          # Direct LLM answer
    ESCALATION = "escalation"              # Human agent needed
    OUT_OF_SCOPE = "out_of_scope"          # Not support-related


class IntentClassifier:
    """
    Classifies user queries to determine workflow routing.
    
    Uses a combination of:
    1. Keyword-based rules (fast, deterministic)
    2. LLM-based classification (for nuanced queries)
    """
    
    def __init__(self):
        self.confidence_threshold = HITL_CONFIDENCE_THRESHOLD
        self.escalation_keywords = [kw.lower() for kw in HITL_ESCALATION_KEYWORDS]
        
        # Initialize free cloud LLM
        try:
            self.llm = get_llm(temperature=0.0, max_tokens=256)
            print(f"🎯 Intent classifier using {self.llm._provider}")
        except Exception as e:
            print(f"⚠️  LLM init failed: {e}")
            self.llm = None
    
    def _keyword_check(self, query: str) -> Dict:
        """
        Fast rule-based intent detection using keywords.
        
        Returns:
            Dict with intent and confidence score
        """
        query_lower = query.lower()
        
        # Check for escalation keywords
        for keyword in self.escalation_keywords:
            if keyword in query_lower:
                return {
                    "intent": IntentType.ESCALATION,
                    "confidence": 0.95,
                    "reason": f"Matched escalation keyword: '{keyword}'",
                    "method": "keyword"
                }
        
        # Check for greetings / casual chat
        chat_patterns = [
            r"^(hi|hello|hey|greetings|howdy)",
            r"^(how are you|what's up|how is it going)",
            r"^(thank|thanks|bye|goodbye)",
            r"^(who are you|what can you do)"
        ]
        for pattern in chat_patterns:
            if re.search(pattern, query_lower):
                return {
                    "intent": IntentType.GENERAL_CHAT,
                    "confidence": 0.9,
                    "reason": "Matched general chat pattern",
                    "method": "keyword"
                }
        
        # Default: assume document query (will be refined by LLM if available)
        return {
            "intent": IntentType.DOCUMENT_QUERY,
            "confidence": 0.5,
            "reason": "Default classification",
            "method": "keyword"
        }
    
    def _llm_classify(self, query: str) -> Dict:
        """
        LLM-based intent classification for nuanced queries.
        
        Returns:
            Dict with intent, confidence, and reasoning
        """
        if self.llm is None:
            return None
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a customer support bot.
Analyze the user query and classify it into ONE of these categories:

1. document_query: Questions about products, policies, accounts, billing, 
   shipping, warranties, returns, technical issues, etc. that can be answered 
   from a knowledge base.

2. general_chat: Greetings, casual conversation, thanks, goodbyes, or questions 
   about the bot's capabilities.

3. escalation: Angry customers, legal threats, fraud reports, account security 
   breaches, complex complaints requiring human judgment, or requests to speak 
   to a human.

4. out_of_scope: Questions completely unrelated to customer support 
   (e.g., weather, politics, personal advice).

Respond ONLY with a JSON object in this exact format:
{{
    "intent": "document_query|general_chat|escalation|out_of_scope",
    "confidence": 0.0-1.0,
    "reason": "Brief explanation of classification"
}}"""),
            ("human", "User query: {query}")
        ])
        
        try:
            # Use JSON mode if Groq is available
            kwargs = {}
            if hasattr(self.llm, '_provider') and self.llm._provider == "groq":
                kwargs["response_format"] = {"type": "json_object"}
            
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"query": query}, config=kwargs)
            
            # Validate intent
            intent_str = result.get("intent", "document_query")
            if intent_str not in [i.value for i in IntentType]:
                intent_str = "document_query"
            
            return {
                "intent": IntentType(intent_str),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", "LLM classification"),
                "method": "llm"
            }
        except Exception as e:
            print(f"⚠️  LLM classification failed: {e}")
            return None
    
    def classify(self, query: str) -> Dict:
        """
        Main classification method combining rules + LLM.
        
        Args:
            query: Raw user query
            
        Returns:
            Dict with intent, confidence, reason, and method used
        """
        print(f"\n🎯 Classifying query: '{query[:80]}...'")
        
        # Step 1: Fast keyword check
        keyword_result = self._keyword_check(query)
        
        # If high-confidence keyword match (escalation or chat), use it
        if keyword_result["intent"] == IntentType.ESCALATION:
            print(f"   → ESCALATION (keyword match): {keyword_result['reason']}")
            return keyword_result
        
        if keyword_result["intent"] == IntentType.GENERAL_CHAT:
            print(f"   → GENERAL_CHAT (keyword match): {keyword_result['reason']}")
            return keyword_result
        
        # Step 2: Refine with LLM for document queries
        llm_result = self._llm_classify(query)
        
        if llm_result and llm_result["confidence"] > keyword_result["confidence"]:
            print(f"   → {llm_result['intent'].upper()} (LLM): {llm_result['reason']}")
            return llm_result
        
        print(f"   → {keyword_result['intent'].upper()}: {keyword_result['reason']}")
        return keyword_result
    
    def should_escalate(self, classification: Dict) -> bool:
        """
        Determine if query should be escalated to human.
        
        Args:
            classification: Result from classify()
            
        Returns:
            True if human intervention needed
        """
        if classification["intent"] == IntentType.ESCALATION:
            return True
        
        # Low confidence in any classification → escalate to be safe
        if classification["confidence"] < self.confidence_threshold:
            return True
        
        return False


# ─── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    classifier = IntentClassifier()
    
    test_queries = [
        "How do I reset my password?",
        "Hello, how are you today?",
        "I want a refund and I'm going to sue your company!",
        "What's the weather like today?",
        "My account was hacked and there are unauthorized charges"
    ]
    
    for query in test_queries:
        result = classifier.classify(query)
        print(f"   Confidence: {result['confidence']:.2f} | Escalate: {classifier.should_escalate(result)}")
        print()