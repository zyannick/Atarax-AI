import mailbox
from email import policy
from email.parser import BytesParser 
from email.message import Message 
from bs4 import BeautifulSoup
from pathlib import Path

def get_text_from_email_payload(message: Message) -> str:
    text_content = ""
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            try:
                body = part.get_payload(decode=True).decode(part.get_content_charset(failobj="utf-8"), errors='replace')
            except Exception:
                body = None

            if body and "attachment" not in content_disposition:
                if content_type == "text/plain":
                    text_content += body + "\n"
                elif content_type == "text/html":
                    soup = BeautifulSoup(body, "html.parser")
                    text_content += soup.get_text(separator="\n") + "\n"
    else: 
        try:
            body = message.get_payload(decode=True).decode(message.get_content_charset(failobj="utf-8"), errors='replace')
            if message.get_content_type() == "text/plain":
                text_content = body
            elif message.get_content_type() == "text/html":
                soup = BeautifulSoup(body, "html.parser")
                text_content = soup.get_text(separator="\n")
        except Exception:
            pass 
    return text_content.strip()

def process_thunderbird_emails(profile_path_str: str):
    profile_path = Path(profile_path_str)
    mail_dirs = ["Mail", "ImapMail"]

    for mail_dir_name in mail_dirs:
        actual_mail_dir = profile_path / mail_dir_name
        if not actual_mail_dir.is_dir():
            continue

        for account_dir in actual_mail_dir.iterdir():
            if not account_dir.is_dir(): 
                continue
            print(f"\nProcessing Account: {account_dir.name}")
            for item in account_dir.rglob('*'):
                try:
                    if item.is_file() and not item.name.endswith(('.msf', '.dat', '.json')) and item.stat().st_size > 0:
                        print(f"  Trying Mbox: {item.name}")
                        try:
                            mb = mailbox.mbox(str(item), factory=None, create=False)
                            for i, msg in enumerate(mb):
                                if i > 5: break # Limiting for example
                                subject = msg['subject']
                                from_ = msg['from']
                                date_ = msg['date']
                                text_body = get_text_from_email_payload(msg)
                                print(f"    Subject: {subject}")
                                print(f"    From: {from_}")
                                print(f"    Date: {date_}")
                                print(f"    Body (plain): {text_body[:200]}...")
                                # TODO : build a more robust text extraction for the RAG system here
                        except Exception as e_mbox:
                            pass 

                    elif item.is_dir() and (item / "cur").is_dir() and (item / "new").is_dir():
                        print(f"  Trying Maildir: {item.name}")
                        try:
                            md = mailbox.Maildir(str(item), factory=None, create=False)
                            for i, msg in enumerate(md):
                                if i > 5: break # Limiting for example
                                subject = msg['subject']
                                from_ = msg['from']
                                date_ = msg['date']
                                text_body = get_text_from_email_payload(msg)
                                print(f"    Subject: {subject}")
                                print(f"    From: {from_}")
                                print(f"    Date: {date_}")
                        except Exception as e_maildir:
                            print(f"    Error processing maildir {item.name}: {e_maildir}")
                except Exception as e_outer:
                    print(f"  Skipping {item.name} due to error: {e_outer}")


