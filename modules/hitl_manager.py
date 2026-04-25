"""
Human-in-the-Loop (HITL) Manager Module
Handles: Escalation tracking, human handoff, and feedback collection
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class EscalationTicket:
    """Represents a ticket escalated to human agents."""
    ticket_id: str
    query: str
    intent: str
    confidence: float
    reason: str
    timestamp: str
    status: str  # "pending", "resolved", "cancelled"
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    customer_satisfaction: Optional[int] = None  # 1-5


class HITLManager:
    """
    Manages Human-in-the-Loop escalation workflow.
    
    Features:
    - Create escalation tickets
    - Track ticket status
    - Simulate human agent responses (for demo)
    - Collect feedback for model improvement
    """
    
    def __init__(self, tickets_file: str = "escalation_tickets.json"):
        self.tickets_file = Path(tickets_file)
        self.tickets: List[EscalationTicket] = []
        self._load_tickets()
    
    def _load_tickets(self):
        """Load existing tickets from file."""
        if self.tickets_file.exists():
            with open(self.tickets_file, "r") as f:
                data = json.load(f)
                self.tickets = [EscalationTicket(**item) for item in data]
            print(f"📋 Loaded {len(self.tickets)} existing tickets")
    
    def _save_tickets(self):
        """Save tickets to file."""
        data = [asdict(t) for t in self.tickets]
        with open(self.tickets_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_ticket(self, query: str, classification: Dict) -> EscalationTicket:
        """
        Create a new escalation ticket.
        
        Args:
            query: Original user query
            classification: Intent classification result
            
        Returns:
            Created EscalationTicket
        """
        ticket = EscalationTicket(
            ticket_id=str(uuid.uuid4())[:8],
            query=query,
            intent=classification["intent"],
            confidence=classification["confidence"],
            reason=classification["reason"],
            timestamp=datetime.now().isoformat(),
            status="pending"
        )
        
        self.tickets.append(ticket)
        self._save_tickets()
        
        print(f"\n🚨 ESCALATION TICKET CREATED")
        print(f"   Ticket ID: {ticket.ticket_id}")
        print(f"   Reason: {ticket.reason}")
        print(f"   Query: {ticket.query[:100]}...")
        
        return ticket
    
    def get_pending_tickets(self) -> List[EscalationTicket]:
        """Get all pending tickets."""
        return [t for t in self.tickets if t.status == "pending"]
    
    def resolve_ticket(self, ticket_id: str, resolution: str, 
                      agent_name: str = "Human Agent") -> Optional[EscalationTicket]:
        """
        Resolve a pending ticket (simulated human agent response).
        
        Args:
            ticket_id: Ticket ID to resolve
            resolution: Human agent's response
            agent_name: Name of the resolving agent
            
        Returns:
            Updated ticket or None if not found
        """
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id and ticket.status == "pending":
                ticket.status = "resolved"
                ticket.resolution = resolution
                ticket.assigned_to = agent_name
                self._save_tickets()
                
                print(f"\n✅ Ticket {ticket_id} resolved by {agent_name}")
                return ticket
        
        return None
    
    def simulate_human_response(self, ticket: EscalationTicket) -> str:
        """
        Simulate a human agent response (for demo purposes).
        In production, this would notify a real human agent.
        
        Args:
            ticket: The escalation ticket
            
        Returns:
            Simulated human response
        """
        responses = {
            "escalation": (
                f"Hello, this is Sarah from TechCorp Support. I've reviewed your "
                f"concern regarding: '{ticket.query}'. I understand your frustration "
                f"and I'm here to help resolve this personally. Please expect a "
                f"follow-up email within 2 hours with next steps. Your ticket "
                f"reference is #{ticket.ticket_id}."
            ),
            "low_confidence": (
                f"Hi, this is Mike from TechCorp Support. Your query "
                f"'{ticket.query}' has been escalated to me as it requires "
                f"specialized attention. I'm looking into this now and will "
                f"get back to you shortly. Ticket: #{ticket.ticket_id}."
            )
        }
        
        if "keyword" in ticket.reason:
            return responses["escalation"]
        return responses["low_confidence"]
    
    def collect_feedback(self, ticket_id: str, rating: int, feedback: str = ""):
        """
        Collect customer feedback on resolution.
        
        Args:
            ticket_id: Ticket ID
            rating: Satisfaction rating (1-5)
            feedback: Optional text feedback
        """
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                ticket.customer_satisfaction = rating
                self._save_tickets()
                print(f"⭐ Feedback recorded for ticket {ticket_id}: {rating}/5")
                return True
        return False
    
    def generate_escalation_response(self, ticket: EscalationTicket) -> Dict:
        """
        Generate the final response sent to user after escalation.
        
        Args:
            ticket: The escalation ticket
            
        Returns:
            Response dict with message and ticket info
        """
        human_response = self.simulate_human_response(ticket)
        
        return {
            "answer": human_response,
            "mode": "hitl",
            "ticket_id": ticket.ticket_id,
            "escalated": True,
            "human_assigned": True,
            "sources": []
        }


# ─── Standalone usage ──────────────────────────────────────────────
if __name__ == "__main__":
    manager = HITLManager()
    
    # Simulate an escalation
    test_classification = {
        "intent": "escalation",
        "confidence": 0.95,
        "reason": "Matched escalation keyword: 'refund'"
    }
    
    ticket = manager.create_ticket(
        "I want a full refund and I'm contacting my lawyer!",
        test_classification
    )
    
    response = manager.generate_escalation_response(ticket)
    print(f"\nResponse to user:\n{response['answer']}")