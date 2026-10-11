alter table public.users add column if not exists approval_notification_status text not null default 'sent';
alter table public.users add column if not exists approval_notification_sent_at timestamptz;
alter table public.users add column if not exists approval_notification_error text;
