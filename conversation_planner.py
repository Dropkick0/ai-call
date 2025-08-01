import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import structlog
from datetime import datetime, timedelta
from groq import Groq
import os
from dotenv import load_dotenv
import calendar_july_2025 as cal
load_dotenv()

logger = structlog.get_logger()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

groq_client = Groq(api_key=GROQ_API_KEY)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

class ConversationPhase(Enum):
    OPENING = "opening"
    RAPPORT_BUILDING = "rapport_building"
    DISCOVERY = "discovery"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"

@dataclass
class Task:
    """Individual conversation task with success criteria"""
    id: str
    name: str
    description: str
    phase: ConversationPhase
    priority: int  # 1-10, higher = more important
    status: TaskStatus = TaskStatus.PENDING
    success_criteria: List[str] = field(default_factory=list)
    completion_indicators: List[str] = field(default_factory=list)
    obstacles: List[str] = field(default_factory=list)
    attempted_strategies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

@dataclass
class Obstacle:
    """Detected conversation obstacle with resolution strategies"""
    type: str
    description: str
    detected_at: datetime
    severity: int  # 1-10
    attempted_solutions: List[str] = field(default_factory=list)
    current_strategy: Optional[str] = None
    resolution_status: str = "active"  # active, resolved, escalated

@dataclass
class ConversationState:
    """Current state of the conversation"""
    current_phase: ConversationPhase = ConversationPhase.OPENING
    active_tasks: List[Task] = field(default_factory=list)
    completed_tasks: List[Task] = field(default_factory=list)
    detected_obstacles: List[Obstacle] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)
    user_signals: Dict[str, Any] = field(default_factory=dict)
    success_indicators: List[str] = field(default_factory=list)
    current_objective: Optional[str] = None
    next_recommended_action: Optional[str] = None
    planning_context: Dict[str, Any] = field(default_factory=dict)

class ConversationPlanner:
    """Strategic conversation planner running in background"""
    
    def __init__(self, conversation_type: str = "cold_call"):
        self.conversation_type = conversation_type
        self.state = ConversationState()
        self.is_planning = False
        self.planning_interval = 3.0  # Run planning every 3 seconds
        self.last_planning = 0
        
        # Initialize conversation checklist based on type
        self._initialize_tasks()
        
        logger.info("Conversation Planner initialized", 
                   conversation_type=conversation_type,
                   total_tasks=len(self.state.active_tasks))
    
    def _initialize_tasks(self):
        """Initialize conversation tasks based on type"""
        if self.conversation_type == "cold_call":
            self.state.active_tasks = [
                Task(
                    id="opening_permission",
                    name="Get Opening Permission",
                    description="Ask for 30 seconds",
                    phase=ConversationPhase.OPENING,
                    priority=10,
                    success_criteria=[
                        "User grants permission to continue",
                        "User doesn't hang up"
                    ],
                    completion_indicators=["yes", "sure", "okay", "go ahead"]
                ),
                Task(
                    id="build_rapport",
                    name="Build Rapport",
                    description="Engage in active listening and small talk",
                    phase=ConversationPhase.RAPPORT_BUILDING,
                    priority=9,
                    success_criteria=[
                        "User engages positively",
                        "Shared some personal/business info"
                    ],
                    completion_indicators=["user_engaged", "information_shared", "positive_tone"]
                ),
                Task(
                    id="transition_pitch",
                    name="Transition to Pitch",
                    description="Ask permission to share idea and present problem/solution",
                    phase=ConversationPhase.PRESENTATION,
                    priority=8,
                    success_criteria=[
                        "User agrees to hear pitch",
                        "Presented value briefly"
                    ],
                    completion_indicators=["pitch_permission", "value_presented"]
                ),
                Task(
                    id="qualify_interest",
                    name="Qualify & Confirm Interest",
                    description="Probe needs and summarize",
                    phase=ConversationPhase.DISCOVERY,
                    priority=7,
                    success_criteria=[
                        "Identified potential interest",
                        "User confirms understanding"
                    ],
                    completion_indicators=["needs_probed", "summary_confirmed"]
                ),
                Task(
                    id="schedule_meeting",
                    name="Schedule Meeting",
                    description="Offer slots, confirm details, send invite",
                    phase=ConversationPhase.CLOSING,
                    priority=10,
                    success_criteria=[
                        "Meeting time agreed",
                        "Details verified"
                    ],
                    completion_indicators=["slot_chosen", "details_confirmed", "invite_sent"]
                ),
                Task(
                    id="closing",
                    name="Close Call",
                    description="Recap, thank, and end positively",
                    phase=ConversationPhase.FOLLOW_UP,
                    priority=6,
                    success_criteria=[
                        "Value recapped",
                        "Positive ending"
                    ],
                    completion_indicators=["recap_given", "thanks_expressed"]
                )
            ]
        
        # Set initial objective
        self._update_current_objective()
    
    def _update_current_objective(self):
        """Update current objective based on active tasks"""
        pending_tasks = [t for t in self.state.active_tasks if t.status == TaskStatus.PENDING]
        
        if pending_tasks:
            # Get highest priority pending task
            current_task = max(pending_tasks, key=lambda t: t.priority)
            self.state.current_objective = f"Focus on: {current_task.name} - {current_task.description}"
            
            # Generate specific action recommendation
            self.state.next_recommended_action = self._get_task_strategy(current_task)

            if current_task and current_task.id == "schedule_meeting":
                # Find two upcoming free slots
                now = datetime.now()
                start = now + timedelta(days=1)  # Start from tomorrow
                free_slots = []
                cur = start.replace(minute=0, second=0, microsecond=0)
                while len(free_slots) < 2 and cur < start + timedelta(days=7):  # Look ahead 1 week
                    if cur in cal.slots and cal.slots[cur] == "free":
                        free_slots.append(cur)
                    cur += timedelta(minutes=30)
                if free_slots:
                    slot1 = free_slots[0].strftime("%A at %I:%M %p")
                    slot2 = free_slots[1].strftime("%A at %I:%M %p") if len(free_slots) > 1 else ''
                    self.state.next_recommended_action += f" Suggest: Would {slot1} or {slot2} work?"
        else:
            self.state.current_objective = "All tasks completed - focus on closing conversation positively"
            self.state.next_recommended_action = "Thank the user and summarize next steps"
    
    def _get_task_strategy(self, task: Task) -> str:
        """Get specific strategy for completing a task"""
        strategies = {
            "opening_permission": "Acknowledge cold call and politely ask for 30 seconds.",
            "build_rapport": "Use active listening and mirror their words to build connection.",
            "transition_pitch": "Ask permission to share and present one-line problem/solution.",
            "qualify_interest": "Probe with one open question and summarize what you heard.",
            "schedule_meeting": "Offer two specific slots and confirm details.",
            "closing": "Recap value briefly and thank them warmly."
        }
        
        base_strategy = strategies.get(task.id, "Focus on completing the task objectives")
        
        # Modify strategy based on obstacles
        if task.obstacles:
            if "resistance" in str(task.obstacles):
                base_strategy += " Address resistance by asking questions and listening more."
            if "time_pressure" in str(task.obstacles):
                base_strategy += " Acknowledge their time constraints and be more concise."
            if "skepticism" in str(task.obstacles):
                base_strategy += " Provide specific examples or social proof to build credibility."
        
        return base_strategy
    
    async def analyze_conversation(self, transcript: List[Dict], user_signals: Dict = None):
        """Analyze conversation progress and update planning"""
        
        if user_signals:
            self.state.user_signals.update(user_signals)
        
        # Update conversation history
        self.state.conversation_history = transcript[-10:]  # Keep last 10 exchanges
        
        # Analyze progress on current tasks
        await self._evaluate_task_progress(transcript)
        
        # Detect obstacles
        await self._detect_obstacles(transcript)
        
        # Update phase if needed
        self._update_conversation_phase()
        
        # Update current objective
        self._update_current_objective()
        
        logger.info("Conversation analyzed",
                   current_phase=self.state.current_phase.value,
                   active_tasks=len([t for t in self.state.active_tasks if t.status == TaskStatus.PENDING]),
                   completed_tasks=len(self.state.completed_tasks),
                   obstacles=len(self.state.detected_obstacles))
    
    async def _evaluate_task_progress(self, transcript: List[Dict]):
        if not transcript:
            return
        
        recent_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in transcript[-5:]])
        
        prompt = f"Analyze snippet for task progress. Active tasks: {', '.join([t.name for t in self.state.active_tasks if t.status != TaskStatus.COMPLETED])}. Score 0-1 per task. Also suggest if any task can be skipped (condition) or if optional task should be added. Output format:\nTASK UPDATES: [task: score, status]\nSKIP SUGGESTION: [task - condition]\nOPTIONAL ADD: [new_task - reason] \n\nSnippet: {recent_text}"
        
        response = self.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",  # Or appropriate model
            messages=[{"role": "system", "content": "You are a concise task analyzer."}, {"role": "user", "content": prompt}],
            temperature=0.3
        )
        analysis = response.choices[0].message.content.strip()
        
        # Parse analysis (simple for now; assume format like 'Task: score')
        for line in analysis.split('\n'):
            if ':' in line:
                task_name, score_str = line.split(':', 1)
                task_name = task_name.strip()
                try:
                    score = float(score_str.strip())
                    task = next((t for t in self.state.active_tasks if t.name == task_name), None)
                    if task:
                        if score >= 0.8:
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = datetime.now()
                            self.state.completed_tasks.append(task)
                            self.state.active_tasks.remove(task)
                            logger.info("Task completed via Groq", task=task.name)
                        elif score >= 0.3:
                            task.status = TaskStatus.IN_PROGRESS
                except:
                    pass
    
    def _check_indicator_in_conversation(self, indicator: str, transcript: List[Dict]) -> bool:
        """Check if completion indicator is present in conversation"""
        recent_text = " ".join([msg.get("content", "") for msg in transcript[-3:]])
        
        indicator_patterns = {
            "opening_permission": ["cold call", "cold", "permission", "30 seconds"],
            "build_rapport": ["tell me", "how", "what", "interested", "positive tone", "small talk"],
            "transition_pitch": ["pitch permission", "value presented", "one-line problem", "one-line solution"],
            "qualify_interest": ["needs probed", "summary confirmed", "open question", "probe"],
            "schedule_meeting": ["slot chosen", "details confirmed", "invite sent", "two specific slots", "offer slots"],
            "closing": ["recap given", "thanks expressed", "value recapped", "positive ending"]
        }
        
        patterns = indicator_patterns.get(indicator, [indicator.lower()])
        return any(pattern in recent_text.lower() for pattern in patterns)
    
    def _detect_task_obstacle(self, task: Task, transcript: List[Dict]) -> bool:
        """Detect if task has hit an obstacle"""
        if not transcript:
            return False
        
        recent_user_messages = [msg.get("content", "") for msg in transcript[-2:] 
                               if msg.get("role") == "user"]
        
        if not recent_user_messages:
            return False
        
        user_text = " ".join(recent_user_messages).lower()
        
        # Common obstacle patterns
        obstacle_patterns = [
            "not interested", "no thanks", "not right now", "busy",
            "don't need", "already have", "not looking", "call back later"
        ]
        
        return any(pattern in user_text for pattern in obstacle_patterns)
    
    async def _detect_obstacles(self, transcript: List[Dict]):
        if not transcript:
            return
        
        user_text = " ".join([msg['content'] for msg in transcript[-3:] if msg['role'] == 'user'])
        
        prompt = f"Detect top obstacle. Output: PRIMARY OBSTACLE: type - brief desc - severity - single strategy"
        
        response = self.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "system", "content": "Output only valid JSON."}, {"role": "user", "content": prompt}],
            temperature=0.3
        )
        try:
            obstacle_data = response.choices[0].message.content.strip()
            if "PRIMARY OBSTACLE:" in obstacle_data:
                obstacle_match = re.search(r"PRIMARY OBSTACLE: (.*?) - (.*?) - (\d+) - (.*?)$", obstacle_data, re.DOTALL)
                if obstacle_match:
                    obstacle_type, description, severity, strategy = obstacle_match.groups()
                    existing = [o for o in self.state.detected_obstacles if o.type == obstacle_type and o.resolution_status == "active"]
                    if not existing:
                        obstacle = Obstacle(
                            type=obstacle_type,
                            description=description,
                            detected_at=datetime.now(),
                            severity=int(severity),
                            current_strategy=strategy
                        )
                        self.state.detected_obstacles.append(obstacle)
                        logger.info("Obstacle detected via Groq", type=obstacle_type)
        except:
            pass
    
    def _add_obstacle(self, obstacle_type: str, description: str, severity: int):
        """Add detected obstacle if not already present"""
        existing = [o for o in self.state.detected_obstacles 
                   if o.type == obstacle_type and o.resolution_status == "active"]
        
        if not existing:
            obstacle = Obstacle(
                type=obstacle_type,
                description=description,
                detected_at=datetime.now(),
                severity=severity
            )
            
            # Suggest resolution strategy
            obstacle.current_strategy = self._get_obstacle_strategy(obstacle_type)
            
            self.state.detected_obstacles.append(obstacle)
            logger.info("Obstacle detected", type=obstacle_type, severity=severity)
    
    def _get_obstacle_strategy(self, obstacle_type: str) -> str:
        """Get strategy for overcoming specific obstacle"""
        strategies = {
            "resistance": "Ask one clarifying question about their concern.",
            "time_pressure": "Acknowledge their time constraints. Offer to schedule a brief follow-up.",
            "skepticism": "Provide specific examples or case studies. Ask what would help them believe.",
            "existing_solution": "Ask about their experience with current solution. Find gaps or improvements."
        }
        
        return strategies.get(obstacle_type, "Acknowledge the concern and ask clarifying questions.")
    
    def _update_conversation_phase(self):
        """Update conversation phase based on completed tasks"""
        completed_ids = [t.id for t in self.state.completed_tasks]
        
        if "schedule_meeting" in completed_ids or "closing" in completed_ids:
            self.state.current_phase = ConversationPhase.FOLLOW_UP
        elif "transition_pitch" in completed_ids:
            self.state.current_phase = ConversationPhase.DISCOVERY
        elif "qualify_interest" in completed_ids:
            self.state.current_phase = ConversationPhase.PRESENTATION
        elif "build_rapport" in completed_ids:
            self.state.current_phase = ConversationPhase.RAPPORT_BUILDING
        elif "opening_permission" in completed_ids:
            self.state.current_phase = ConversationPhase.OPENING
    
    def get_current_guidance(self) -> Dict[str, Any]:
        """Get current guidance for the speaking AI"""
        active_obstacles = [o for o in self.state.detected_obstacles if o.resolution_status == "active"]
        pending_tasks = [t for t in self.state.active_tasks if t.status == TaskStatus.PENDING]
        
        guidance = {
            "current_phase": self.state.current_phase.value,
            "current_objective": self.state.current_objective,
            "recommended_action": self.state.next_recommended_action,
            "active_obstacles": [
                {
                    "type": o.type,
                    "description": o.description,
                    "strategy": o.current_strategy,
                    "severity": o.severity
                } for o in active_obstacles
            ],
            "next_tasks": [
                {
                    "name": t.name,
                    "description": t.description,
                    "priority": t.priority,
                    "success_criteria": t.success_criteria
                } for t in pending_tasks[:2]  # Next 2 tasks
            ],
            "conversation_progress": {
                "completed_tasks": len(self.state.completed_tasks),
                "total_tasks": len(self.state.active_tasks) + len(self.state.completed_tasks),
                "success_rate": len(self.state.completed_tasks) / (len(self.state.active_tasks) + len(self.state.completed_tasks)) if self.state.active_tasks or self.state.completed_tasks else 0
            }
        }
        
        # Ensure we always return valid guidance
        if not guidance.get("recommended_action"):
            guidance["recommended_action"] = "continue_conversation"
            
        return guidance
    
    async def start_background_planning(self, transcript_provider, interval: float = 3.0):
        """Start background planning loop"""
        self.is_planning = True
        self.planning_interval = interval
        
        logger.info("Background planning started", interval=interval)
        
        while self.is_planning:
            try:
                # Get latest transcript
                transcript = await transcript_provider.get_conversation_history()
                
                # Analyze and update planning
                await self.analyze_conversation(transcript)
                
                # Wait for next cycle
                await asyncio.sleep(self.planning_interval)
                
            except Exception as e:
                logger.error("Planning cycle error", error=str(e))
                await asyncio.sleep(1)  # Brief pause before retry
    
    def stop_planning(self):
        """Stop background planning"""
        self.is_planning = False
        logger.info("Background planning stopped")
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        completed = len(self.state.completed_tasks)
        total = len(self.state.active_tasks) + completed
        progress = f"{completed}/{total} tasks completed"
        
        phase = self.state.current_phase.value.replace("_", " ").title()
        
        obstacles = len([o for o in self.state.detected_obstacles if o.resolution_status == "active"])
        obstacle_text = f" | {obstacles} active obstacles" if obstacles else ""
        
        return f"Phase: {phase} | Progress: {progress}{obstacle_text}" 