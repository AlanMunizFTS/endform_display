import paramiko
from settings import get_sftp_settings
from paths_config import REMOTE_HIST_DISPLAY_DIR

# ===== CONFIGURA AQUÍ TUS VARIABLES =====
_sftp_settings = get_sftp_settings()
hostname = _sftp_settings["hostname"]
port = _sftp_settings["port"]
username = _sftp_settings["username"]
password = _sftp_settings["password"]
remote_hist_path = REMOTE_HIST_DISPLAY_DIR
# ========================================

print("Conectando al servidor SFTP...")

try:
    # Crear cliente SSH
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Conectar
    ssh_client.connect(
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        timeout=10
    )
    
    # Abrir sesión SFTP
    sftp_client = ssh_client.open_sftp()
    
    print(f"✓ Conexión exitosa!\n")
    print(f"Contenido de {remote_hist_path}:")
    print("=" * 60)
    
    try:
        # Listar archivos en hist_display
        files = sftp_client.listdir(remote_hist_path)
        
        if not files:
            print("(Carpeta vacía)")
        else:
            for idx, file in enumerate(files, 1):
                # Obtener información del archivo
                file_path = f"{remote_hist_path}/{file}"
                stat = sftp_client.stat(file_path)
                size_kb = stat.st_size / 1024
                print(f"{idx}. {file} ({size_kb:.2f} KB)")
        
        print("=" * 60)
        print(f"Total de archivos: {len(files)}")
        
    except FileNotFoundError:
        print(f"✗ La carpeta {remote_hist_path} no existe en el servidor")
        print("  (Se creará automáticamente cuando guardes imágenes por primera vez)")
    
    # Cerrar conexión
    sftp_client.close()
    ssh_client.close()
    print("\n✓ Conexión cerrada")
    
except paramiko.AuthenticationException:
    print("✗ Error: Autenticación fallida")
except paramiko.SSHException as e:
    print(f"✗ Error SSH: {str(e)}")
except Exception as e:
    print(f"✗ Error: {str(e)}")
