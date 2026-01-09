# FYP_Workbench/user_manager.py
import json
import os
from dataclasses import dataclass, asdict

# --- FIX: USE ABSOLUTE PATH ---
# This ensures the DB is always in FYP_Workbench/, no matter where you run the terminal command.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DB_FILE = os.path.join(BASE_DIR, "users_db.json")


@dataclass
class User:
    username: str
    password: str  # In production, hash this!
    role: str  # 'Staff', 'Admin', 'Master Admin'


class UserManager:
    def __init__(self):
        self.users = self._load_users()

    def _load_users(self):
        if not os.path.exists(USER_DB_FILE):
            # Create default Master Admin if db doesn't exist
            defaults = {"master": {"username": "master", "password": "123", "role": "Master Admin"}}
            self._save_users_to_file(defaults)
            return defaults
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)

    def _save_users_to_file(self, data):
        with open(USER_DB_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def login(self, username, password):
        # FIX: Strip spaces so "master " works as "master"
        clean_user = username.strip()
        clean_pass = password.strip()

        user_data = self.users.get(clean_user)
        if user_data and user_data['password'] == clean_pass:
            return User(**user_data)
        return None

    def register_user(self, creator_role, new_username, new_password, new_role):
        # Rule: Only Admin and Master Admin can register users
        if creator_role not in ["Admin", "Master Admin"]:
            raise PermissionError("Only Admins can register new users.")

        if new_username in self.users:
            raise ValueError("Username already exists.")

        self.users[new_username] = {
            "username": new_username,
            "password": new_password,
            "role": new_role
        }
        self._save_users_to_file(self.users)
        return f"User '{new_username}' created successfully."

    def delete_user(self, creator_role, target_username):
        # Rule: Only Master Admin can delete users
        if creator_role != "Master Admin":
            raise PermissionError("Only Master Admin can delete users.")

        if target_username not in self.users:
            raise ValueError("User not found.")

        del self.users[target_username]
        self._save_users_to_file(self.users)
        return f"User '{target_username}' deleted."

    def update_role(self, creator_role, target_username, new_role):
        # Rule: Only Master Admin can update roles
        if creator_role != "Master Admin":
            raise PermissionError("Only Master Admin can update user roles.")

        if target_username not in self.users:
            raise ValueError("User not found.")

        self.users[target_username]['role'] = new_role
        self._save_users_to_file(self.users)
        return f"User '{target_username}' is now a {new_role}."