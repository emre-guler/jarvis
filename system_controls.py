import os
import subprocess
import platform

class SystemControls:
    def __init__(self):
        self.system = platform.system()
        
    def execute_command(self, command):
        """Execute a system command safely"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"Error executing command: {e}")
            return None
    
    def shutdown_computer(self):
        """Shutdown the computer"""
        if self.system == "Darwin":  # macOS
            os.system("sudo shutdown -h now")
        elif self.system == "Windows":
            os.system("shutdown /s /t 0")
        elif self.system == "Linux":
            os.system("sudo shutdown -h now")
    
    def restart_computer(self):
        """Restart the computer"""
        if self.system == "Darwin":  # macOS
            os.system("sudo shutdown -r now")
        elif self.system == "Windows":
            os.system("shutdown /r /t 0")
        elif self.system == "Linux":
            os.system("sudo shutdown -r now")
    
    def sleep_computer(self):
        """Put the computer to sleep"""
        if self.system == "Darwin":  # macOS
            os.system("pmset sleepnow")
        elif self.system == "Windows":
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif self.system == "Linux":
            os.system("systemctl suspend")
    
    def control_screen_brightness(self, level):
        """Control screen brightness (0-100)"""
        if self.system == "Darwin":
            # Convert 0-100 to 0.0-1.0
            normalized_level = level / 100.0
            cmd = f"brightness {normalized_level}"
            self.execute_command(cmd)
    
    def control_volume(self, level):
        """Control system volume (0-100)"""
        if self.system == "Darwin":
            os.system(f"osascript -e 'set volume output volume {level}'")
    
    def mute_volume(self):
        """Mute system volume"""
        if self.system == "Darwin":
            os.system("osascript -e 'set volume output muted true'")
    
    def unmute_volume(self):
        """Unmute system volume"""
        if self.system == "Darwin":
            os.system("osascript -e 'set volume output muted false'")
    
    def open_application(self, app_name):
        """Open an application"""
        if self.system == "Darwin":
            os.system(f"open -a '{app_name}'")
    
    def close_application(self, app_name):
        """Close an application"""
        if self.system == "Darwin":
            os.system(f"osascript -e 'quit app \"{app_name}\"'")
    
    def get_battery_status(self):
        """Get battery status"""
        if self.system == "Darwin":
            try:
                # Get power source
                power_cmd = "pmset -g ps"
                power_result = self.execute_command(power_cmd)
                power_source = "Şarj ediliyor" if "AC Power" in power_result else "Pil gücü kullanılıyor"
                
                # Get battery percentage
                percent_cmd = "pmset -g batt | grep -Eo '\\d+%'"
                percent_result = self.execute_command(percent_cmd)
                percentage = percent_result.strip() if percent_result else "Bilinmiyor"
                
                # Get time remaining
                time_cmd = "pmset -g batt | grep -Eo '\\d+:\\d+ remaining'"
                time_result = self.execute_command(time_cmd)
                
                status = f"{power_source}, {percentage}"
                
                if time_result:
                    time_parts = time_result.split(':')[0]
                    if time_parts and time_parts.strip().isdigit():
                        hours = int(time_parts)
                        if hours > 0:
                            status += f", yaklaşık {hours} saat kaldı"
                
                return status
                
            except Exception as e:
                print(f"Pil durumu ayrıştırılırken hata: {e}")
                return "Pil durumu alınamadı" 