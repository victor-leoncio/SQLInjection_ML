from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import logging
import json
from typing import Set
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

blocked_ips: Set[str] = set()

class BlockIPRequest(BaseModel):
    ip: str
    reason: str = "SQL Injection detected"

def execute_iptables_command(command: list) -> dict:
    """Execute iptables command safely"""
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command executed successfully: {' '.join(command)}")
        return {"success": True, "output": result.stdout}
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {' '.join(command)} - Error: {e.stderr}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

def block_ip_iptables(ip: str) -> dict:
    """Block IP using iptables"""
    if ip in blocked_ips:
        return {"success": True, "message": f"IP {ip} already blocked"}
    
    command = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
    result = execute_iptables_command(command)
    
    if result["success"]:
        blocked_ips.add(ip)
        logger.info(f"IP {ip} successfully blocked")
        return {"success": True, "message": f"IP {ip} blocked successfully"}
    else:
        return {"success": False, "error": result["error"]}

def unblock_ip_iptables(ip: str) -> dict:
    """Unblock IP using iptables"""
    if ip not in blocked_ips:
        return {"success": True, "message": f"IP {ip} is not blocked"}
    
    command = ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]
    result = execute_iptables_command(command)
    
    if result["success"]:
        blocked_ips.discard(ip)
        logger.info(f"IP {ip} successfully unblocked")
        return {"success": True, "message": f"IP {ip} unblocked successfully"}
    else:
        return {"success": False, "error": result["error"]}

@app.post("/block-ip")
async def block_ip(request: BlockIPRequest):
    """Endpoint to block an IP address"""
    ip = request.ip
    reason = request.reason
    
    logger.info(f"Received request to block IP: {ip}, Reason: {reason}")
    
    if not ip:
        raise HTTPException(status_code=400, detail="IP address is required")
    
    result = block_ip_iptables(ip)
    
    if result["success"]:
        return {
            "status": "success",
            "message": result["message"],
            "blocked_ip": ip,
            "reason": reason,
            "total_blocked": len(blocked_ips)
        }
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.post("/unblock-ip")
async def unblock_ip(request: BlockIPRequest):
    """Endpoint to unblock an IP address"""
    ip = request.ip
    
    logger.info(f"Received request to unblock IP: {ip}")
    
    if not ip:
        raise HTTPException(status_code=400, detail="IP address is required")
    
    result = unblock_ip_iptables(ip)
    
    if result["success"]:
        return {
            "status": "success",
            "message": result["message"],
            "unblocked_ip": ip,
            "total_blocked": len(blocked_ips)
        }
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/blocked-ips")
async def get_blocked_ips():
    """Get list of currently blocked IPs"""
    return {
        "blocked_ips": list(blocked_ips),
        "total_blocked": len(blocked_ips)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "firewall-manager",
        "blocked_ips_count": len(blocked_ips)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize iptables rules on startup"""
    logger.info("Firewall Manager starting up...")
    
    result = execute_iptables_command(["iptables", "--version"])
    if not result["success"]:
        logger.error("iptables is not available!")
    else:
        logger.info("iptables is available and ready")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
