#pragma once
#include <functional>
#include <windows.h>
#include "base.h"


enum EventFlags {
	EF_NONE = 0,
	EF_SHIFT_DOWN = 1 << 0,
	EF_CONTROL_DOWN = 1 << 1,
	EF_ALT_DOWN = 1 << 2,
	EF_LEFT_BUTTON_DOWN = 1 << 3,
	EF_MIDDLE_BUTTON_DOWN = 1 << 4,
	EF_RIGHT_BUTTON_DOWN = 1 << 5
};

struct FreeCameraModel {
	enum class CameraMoveType : uint8_t {
		NONE,
		TUMBLE,
		TRACK,
		DOLLY,
	};
	FreeCameraModel() = default;
	CameraMoveType move_type{ CameraMoveType::NONE };
	Vec2i mouse_pos;
	float theta{ 0 };
	float phi{ 0 };
	Vec3<float> look_at{ 0 };
	float distance_to_target{ 5 };
	Quat<float> camera_rotation;
};

class Window {
	using renderFunc = std::function<float(const DensityGrid& grid, const RenderSetting& setting, unsigned char* frameBuffer)>;
public:
	Window(int width = 640, int height = 480);
	~Window();

	void Run();

	void SetRenderSetting(RenderSetting* setting) {
		m_renderSetting = setting;
	}

	void SetDensityGrid(DensityGrid* densityGrid) {
		m_densityGrid = densityGrid;
	}

	void SetRenderCallback(renderFunc callback) {
		m_renderCallback = std::move(callback);
	}

	void Render();

private:
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

	void CreateAndRegisterWindow(HINSTANCE hInstance);

	void OnMouseReleased();
	void OnMousePressed(int flags, Vec2i location);
	void OnMouseMoved(const Vec2i& location);
	void OnMouseWheel(int scroll_amount);

	void SetCameraMatrices();
	void UpdateCameraRotation();

private:
	const wchar_t* CLASSNAME = L"Window";
	HWND hwnd;
	HDC hdcBuffer;
	void* pvBits; // Pointer to the bitmap's pixel bits
	HBITMAP hbmBuffer;

	int m_width;
	int m_height;
	float fps = 0.0f;

	RenderSetting* m_renderSetting = nullptr;
	DensityGrid* m_densityGrid = nullptr;
	renderFunc m_renderCallback;


	FreeCameraModel freecam_model;
	Matrix44<float> rotation_mat;
	float kRotateAmplitude = 0.01f;
	float kPanAmplitude = 0.01f;
	float kScrollAmplitude = 0.1f;

	Matrix44<float> cam_to_world;
}; // 类定义结束