import os
import re
import csv
import glob
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Attempt optional library imports with graceful fallbacks
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import whisper
except ImportError:
    whisper = None

# =====================================================================
# DATA ENGINE & GRAPH CONTEXT LOADER
# =====================================================================

class DatasetContext:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.users: Dict[str, Dict] = {}
        self.groups: Dict[str, Dict] = {}
        self.group_members: Dict[Tuple[str, str], Dict] = {} # (user_id, group_id) -> data
        self.business_accounts: Dict[str, Dict] = {}
        self.user_business_history: Dict[Tuple[str, str], Dict] = {} # (user_id, business_id) -> data
        self.images: Dict[str, str] = {} # image_id -> file_path
        self.voice_notes: Dict[str, str] = {} # voice_note_id -> file_path
        self.message_history: Dict[str, Dict] = {} # message_id -> history dict
        self.user_message_history: Dict[str, List[Dict]] = {} # user_id -> list of past messages
        
        self.load_all()

    def _read_csv(self, filename: str) -> List[Dict[str, str]]:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return []
        with open(path, mode='r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def load_all(self):
        # Load Users
        for row in self._read_csv("users.csv"):
            self.users[row['user_id']] = row

        # Load Groups
        for row in self._read_csv("groups.csv"):
            self.groups[row['group_id']] = row

        # Load Group Members
        for row in self._read_csv("group_members.csv"):
            key = (row['user_id'], row['group_id'])
            self.group_members[key] = row

        # Load Business Accounts
        for row in self._read_csv("business_accounts.csv"):
            self.business_accounts[row['business_id']] = row

        # Load User Business History
        for row in self._read_csv("user_business_history.csv"):
            key = (row['user_id'], row['business_id'])
            self.user_business_history[key] = row

        # Load Images mapping
        for row in self._read_csv("images.csv"):
            self.images[row['image_id']] = row.get('file_path', '')

        # Load Voice Notes mapping
        for row in self._read_csv("voice_notes.csv"):
            self.voice_notes[row['voice_note_id']] = row.get('file_path', '')

        # Load Message History
        for row in self._read_csv("message_history.csv"):
            msg_id = row['message_id']
            u_id = row.get('user_id', '')
            self.message_history[msg_id] = row
            if u_id:
                if u_id not in self.user_message_history:
                    self.user_message_history[u_id] = []
                self.user_message_history[u_id].append(row)

# =====================================================================
# MULTIMODAL FEATURE EXTRACTOR
# =====================================================================

class MultimodalProcessor:
    def __init__(self, context: DatasetContext):
        self.context = context
        self.whisper_model = None
        # Initialize Whisper if available
        if whisper:
            try:
                self.whisper_model = whisper.load_model("tiny")
            except Exception:
                self.whisper_model = None

    def process_message_content(self, msg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extracts full text representation and multimodal attributes.
        Returns: (extracted_text, metadata_dict)
        """
        media_type = msg.get('media_type', '').strip().lower()
        media_id = msg.get('media_id', '').strip()
        raw_text = msg.get('message_text', '').strip()
        
        extracted_text = raw_text
        meta = {'is_ocr': False, 'is_transcript': False, 'has_media': False}

        # Handle Image Messages
        if media_type == 'image' and media_id:
            meta['has_media'] = True
            image_path = self.context.images.get(media_id, '')
            full_img_path = os.path.join(self.context.data_dir, image_path)
            
            ocr_text = ""
            if os.path.exists(full_img_path) and pytesseract:
                try:
                    img = Image.open(full_img_path)
                    ocr_text = pytesseract.image_to_string(img).strip()
                    meta['is_ocr'] = True
                except Exception:
                    ocr_text = ""
            
            if ocr_text:
                extracted_text = f"{raw_text} [IMAGE OCR: {ocr_text}]".strip()

        # Handle Voice Note Messages
        elif media_type == 'voice' and media_id:
            meta['has_media'] = True
            voice_path = self.context.voice_notes.get(media_id, '')
            full_voice_path = os.path.join(self.context.data_dir, voice_path)
            
            transcript = ""
            if os.path.exists(full_voice_path) and self.whisper_model:
                try:
                    res = self.whisper_model.transcribe(full_voice_path)
                    transcript = res.get('text', '').strip()
                    meta['is_transcript'] = True
                except Exception:
                    transcript = ""
            
            if transcript:
                extracted_text = f"[VOICE TRANSCRIPT: {transcript}]".strip()

        return extracted_text, meta

# =====================================================================
# HISTORICAL EVIDENCE & RETRIEVAL ENGINE
# =====================================================================

class EvidenceRetriever:
    def __init__(self, context: DatasetContext):
        self.context = context

    def find_evidence(self, user_id: str, current_text: str, sender_id: str, conversation_type: str) -> List[str]:
        """
        Finds historical message IDs that share semantic similarity or repeat interaction patterns.
        """
        past_msgs = self.context.user_message_history.get(user_id, [])
        if not past_msgs:
            return []

        tokens = set(re.findall(r'\w+', current_text.lower()))
        if not tokens:
            return []

        matched_ids = []
        for past in past_msgs:
            # Check conversation/sender match
            past_sender = past.get('sender_user_id', '') or past.get('business_id', '')
            if past_sender and past_sender == sender_id:
                past_text = past.get('message_text', '').lower()
                past_tokens = set(re.findall(r'\w+', past_text))
                
                # Jaccard overlap
                intersection = tokens.intersection(past_tokens)
                if len(intersection) >= 2 or (len(tokens) > 0 and len(intersection) / len(tokens) > 0.4):
                    matched_ids.append(past.get('message_id'))
                    if len(matched_ids) >= 3:
                        break

        return matched_ids

# =====================================================================
# MESSAGE NOTIFICATION ROUTING SYSTEM
# =====================================================================

class MessageRouter:
    def __init__(self, context: DatasetContext):
        self.context = context
        self.processor = MultimodalProcessor(context)
        self.retriever = EvidenceRetriever(context)

    def is_in_quiet_hours(self, user_id: str, timestamp_str: str) -> bool:
        user = self.context.users.get(user_id, {})
        quiet_hours = user.get('quiet_hours', '') # e.g., "22:00-07:00"
        if not quiet_hours or '-' not in quiet_hours:
            return False

        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            cur_time = dt.time()
            start_str, end_str = quiet_hours.split('-')
            start_t = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_t = datetime.strptime(end_str.strip(), "%H:%M").time()

            if start_t <= end_t:
                return start_t <= cur_time <= end_t
            else: # Overnight quiet hours
                return cur_time >= start_t or cur_time <= end_t
        except Exception:
            return False

    def route_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = msg['message_id']
        user_id = msg['user_id']
        conv_type = msg.get('conversation_type', 'personal')
        group_id = msg.get('group_id', '')
        biz_id = msg.get('business_id', '')
        sender_id = msg.get('sender_user_id', '') or biz_id
        created_at = msg.get('created_at', '')
        forwarded_count = int(msg.get('forwarded_count', 0) or 0)

        # 1. Process Multimodal Content
        extracted_text, media_meta = self.processor.process_message_content(msg)
        lower_text = extracted_text.lower()

        # 2. Retrieve Historical Evidence
        evidence_ids = self.retriever.find_evidence(user_id, extracted_text, sender_id, conv_type)
        evidence_str = ";".join(evidence_ids) if evidence_ids else "none"

        # 3. Security & Scam Risk Assessment (Hard Overrides)
        scam_keywords = ['lottery', 'won cash', 'bank details', 'verify account', 'click link', 'crypto bonus', 'urgent wire']
        is_unverified_biz = False
        biz_data = self.context.business_accounts.get(biz_id, {})
        if biz_id and biz_data:
            is_verified = biz_data.get('is_verified', 'false').lower() == 'true'
            report_rate = float(biz_data.get('report_rate', 0.0) or 0.0)
            if not is_verified or report_rate > 0.05:
                is_unverified_biz = True

        contains_phishing = any(kw in lower_text for kw in scam_keywords)
        
        if (conv_type == 'business' and is_unverified_biz and contains_phishing) or (forwarded_count > 5 and contains_phishing):
            return {
                'message_id': msg_id,
                'action': 'mute',
                'message_type': 'scam',
                'reason': 'Suspicious sender and unverified links flagged for potential phishing or scam.',
                'confidence': 0.96,
                'evidence_message_ids': evidence_str
            }

        if forwarded_count >= 10 and any(kw in lower_text for kw in ['good morning', 'forwarded', 'share to 10 friends']):
            return {
                'message_id': msg_id,
                'action': 'mute',
                'message_type': 'spam',
                'reason': 'Excessively forwarded viral chain message suppressed as repetitive spam.',
                'confidence': 0.92,
                'evidence_message_ids': evidence_str
            }

        # 4. Contextual Checks
        in_quiet_hours = self.is_in_quiet_hours(user_id, created_at)
        
        # Check Direct Mention in Group
        user_data = self.context.users.get(user_id, {})
        user_name = user_data.get('name', '').lower()
        is_direct_mention = False
        if conv_type == 'group' and user_name:
            if f"@{user_name}" in lower_text or f"@{user_id.lower()}" in lower_text:
                is_direct_mention = True

        # Check Group Settings
        group_muted = False
        if conv_type == 'group' and group_id:
            member_info = self.context.group_members.get((user_id, group_id), {})
            group_muted = member_info.get('is_muted', 'false').lower() == 'true'

        # Check Urgency / Category Keywords
        urgent_keywords = ['urgent', 'emergency', 'hospital', 'otp', 'code', 'cancelled', 'immediately', 'help', 'deadline']
        is_urgent = any(kw in lower_text for kw in urgent_keywords)

        payment_keywords = ['receipt', 'invoice', 'paid', 'payment due', 'transaction', 'upi', 'bank alert']
        is_payment = any(kw in lower_text for kw in payment_keywords)

        promo_keywords = ['off', 'discount', 'sale', 'deal', 'buy 1 get 1', 'coupon', 'limited time offer']
        is_promo = any(kw in lower_text for kw in promo_keywords)

        greeting_keywords = ['good morning', 'good night', 'happy birthday', 'congratulations', 'festive greetings']
        is_greeting = any(kw in lower_text for kw in greeting_keywords)

        # 5. Routing Decision Logic
        
        # Priority 1: Direct Mention or Explicit High Urgency (Bypasses group mute and quiet hours if critical)
        if is_direct_mention or (is_urgent and conv_type == 'personal'):
            msg_type = 'urgent' if is_urgent else 'personal'
            return {
                'message_id': msg_id,
                'action': 'notify',
                'message_type': msg_type,
                'reason': f"Immediate attention required: {'Direct mention in group' if is_direct_mention else 'Time-sensitive urgent message'}.",
                'confidence': 0.95,
                'evidence_message_ids': evidence_str
            }

        # Priority 2: Payment Notifications from Legitimate Senders
        if is_payment:
            biz_hist = self.context.user_business_history.get((user_id, biz_id), {})
            has_recent_orders = biz_hist.get('has_active_orders', 'false').lower() == 'true'
            if conv_type == 'business' and (not is_unverified_biz or has_recent_orders):
                return {
                    'message_id': msg_id,
                    'action': 'notify',
                    'message_type': 'payment',
                    'reason': 'Important financial payment update from verified business.',
                    'confidence': 0.91,
                    'evidence_message_ids': evidence_str
                }
            else:
                return {
                    'message_id': msg_id,
                    'action': 'digest',
                    'message_type': 'payment',
                    'reason': 'Payment notice held for review batch digest.',
                    'confidence': 0.78,
                    'evidence_message_ids': evidence_str
                }

        # Priority 3: Quiet Hours Enforcement
        if in_quiet_hours:
            return {
                'message_id': msg_id,
                'action': 'digest',
                'message_type': 'business_update' if conv_type == 'business' else 'personal',
                'reason': 'Non-urgent message routed to digest during user quiet hours.',
                'confidence': 0.88,
                'evidence_message_ids': evidence_str
            }

        # Priority 4: Muted Group Filtering
        if conv_type == 'group' and group_muted:
            return {
                'message_id': msg_id,
                'action': 'digest' if not is_greeting else 'mute',
                'message_type': 'event' if 'meeting' in lower_text or 'event' in lower_text else 'greeting' if is_greeting else 'personal',
                'reason': 'Message from explicitly muted group batched without immediate alert.',
                'confidence': 0.85,
                'evidence_message_ids': evidence_str
            }

        # Priority 5: Promotional & Greeting Content
        if is_promo:
            # Check user history with business
            biz_hist = self.context.user_business_history.get((user_id, biz_id), {})
            opted_in = biz_hist.get('opted_in', 'false').lower() == 'true'
            
            if opted_in:
                return {
                    'message_id': msg_id,
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': 'Opted-in promotional update batched into promotional digest.',
                    'confidence': 0.86,
                    'evidence_message_ids': evidence_str
                }
            else:
                return {
                    'message_id': msg_id,
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': 'Unsolicited promotional message muted to reduce user interruption.',
                    'confidence': 0.89,
                    'evidence_message_ids': evidence_str
                }

        if is_greeting and forwarded_count > 1:
            return {
                'message_id': msg_id,
                'action': 'mute',
                'message_type': 'greeting',
                'reason': 'Repetitive broadcast greeting muted.',
                'confidence': 0.84,
                'evidence_message_ids': evidence_str
            }

        # Priority 6: Standard Personal Messages
        if conv_type == 'personal':
            return {
                'message_id': msg_id,
                'action': 'notify',
                'message_type': 'personal',
                'reason': 'Direct 1-on-1 personal conversation.',
                'confidence': 0.90,
                'evidence_message_ids': evidence_str
            }

        # Default Catch-all (Digest)
        return {
            'message_id': msg_id,
            'action': 'digest',
            'message_type': 'business_update' if conv_type == 'business' else 'personal',
            'reason': 'General message categorized for periodic notification summary digest.',
            'confidence': 0.75,
            'evidence_message_ids': evidence_str
        }

# =====================================================================
# MAIN PIPELINE RUNNER
# =====================================================================

def run_pipeline(dataset_dir: str = "dataset", output_file: str = "output.csv"):
    print(f"[*] Initializing WhatsApp Notification Router Engine from {dataset_dir}...")
    context = DatasetContext(dataset_dir)
    router = MessageRouter(context)

    messages_path = os.path.join(dataset_dir, "messages.csv")
    if not os.path.exists(messages_path):
        print(f"[!] Error: {messages_path} not found.")
        return

    predictions = []
    with open(messages_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            res = router.route_message(row)
            predictions.append(res)

    # Write predictions to output.csv
    fieldnames = ['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"[+] Successfully processed {len(predictions)} messages and generated {output_file}.")

if __name__ == "__main__":
    run_pipeline()
