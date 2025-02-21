#include "window.h"
#include <stdexcept>
#include <cassert>

#include <windowsx.h> // GET_X_LPARAM
#include <string>


Window::Window(int width, int height)
	:m_width(width), m_height(height)
{
	HINSTANCE hInstance = GetModuleHandle(NULL);
	CreateAndRegisterWindow(hInstance);

	// 初始化相机参数
	cam_to_world[0][3] = 0;
	cam_to_world[1][3] = 0;
	cam_to_world[2][3] = 10;
}

Window::~Window()
{
}

void Window::Run()
{
	MSG msg;
	while (1) {
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE) != 0) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			if (msg.message == WM_QUIT) {
				break;
			}
		}
		if (msg.message == WM_QUIT)
			break;
	}
}

LRESULT CALLBACK Window::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	Window* pThis = nullptr;
	pThis = reinterpret_cast<Window*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
	if (!pThis)
		return DefWindowProc(hWnd, msg, wParam, lParam);

	switch (msg) {
	case WM_CLOSE:
		DeleteObject(pThis->hbmBuffer);
		DestroyWindow(pThis->hwnd);
		break;
	case WM_CREATE:
		pThis->Render();
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_LBUTTONUP:
	case WM_MBUTTONUP:
	case WM_RBUTTONUP:
		pThis->OnMouseReleased();
		InvalidateRect(pThis->hwnd, NULL, TRUE);
		UpdateWindow(pThis->hwnd);
		break;
	case WM_LBUTTONDOWN:
	case WM_MBUTTONDOWN:
	case WM_RBUTTONDOWN: {
		Vec2i location(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));

		unsigned int flags = 0;
		if (GetKeyState(VK_SHIFT) & 0x8000) flags |= EF_SHIFT_DOWN;
		if (GetKeyState(VK_CONTROL) & 0x8000) flags |= EF_CONTROL_DOWN;
		if (GetKeyState(VK_MENU) & 0x8000) flags |= EF_ALT_DOWN; // VK_MENU is the Alt key
		if (wParam & MK_LBUTTON) flags |= EF_LEFT_BUTTON_DOWN;
		if (wParam & MK_MBUTTON) flags |= EF_MIDDLE_BUTTON_DOWN;
		if (wParam & MK_RBUTTON) flags |= EF_RIGHT_BUTTON_DOWN;

		pThis->OnMousePressed(flags, location);
	}
					   break;
	case WM_MOUSEMOVE: {
		int xpos = GET_X_LPARAM(lParam);
		int ypos = GET_Y_LPARAM(lParam);
		pThis->OnMouseMoved(Vec2i(xpos, ypos));
	}
					 break;
	case WM_MOUSEWHEEL: {
		int delta = GET_WHEEL_DELTA_WPARAM(wParam) / WHEEL_DELTA;
		pThis->OnMouseWheel(delta);
	}
					  break;
	case WM_ERASEBKGND:
		return 1; // Indicate that background erase is handled
	case WM_PAINT: {
		const std::wstring mode[4] = { L"None",
									   L"Tumble (ALT+LMB)",
									   L"Track (Alt+MMB)",
									   L"Dolly (ALT+RMB/Wheel)" };
		PAINTSTRUCT ps;
		HDC hdcWindow = BeginPaint(pThis->hwnd, &ps);
		BitBlt(hdcWindow, 0, 0, pThis->m_width, pThis->m_height, pThis->hdcBuffer, 0, 0, SRCCOPY);
		std::wstring text = L"fps: " + std::to_wstring(pThis->fps);
		SetTextColor(hdcWindow, RGB(255, 255, 255)); // White text
		SetBkMode(hdcWindow, TRANSPARENT);
		TextOut(hdcWindow, 10, 10, text.c_str(), text.length());
		TextOut(hdcWindow, 10, 28, mode[(int)pThis->freecam_model.move_type].c_str(),
			mode[(int)pThis->freecam_model.move_type].length());
	} break;
	default:
		return DefWindowProc(hWnd, msg, wParam, lParam);
	}
	return 0;
}

void Window::CreateAndRegisterWindow(HINSTANCE hInstance)
{
	WNDCLASSEX wc = {};
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.lpfnWndProc = WndProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASSNAME;
	wc.hCursor = LoadCursor(nullptr, IDC_ARROW); // Set the default arrow cursor
	wc.hIcon = LoadIcon(hInstance, IDI_APPLICATION); // Load the default application icon
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wc.lpszMenuName = nullptr;
	wc.hIconSm = LoadIcon(hInstance, IDI_APPLICATION); // Load the small icon for the application

	if (!RegisterClassEx(&wc)) {
		MessageBox(nullptr, L"Window Registration Failed", L"Error",
			MB_ICONEXCLAMATION | MB_OK);
	}

	hwnd = CreateWindowEx(
		WS_EX_CLIENTEDGE,
		CLASSNAME,
		L"3D Navigation Controls",
		WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX, // non-resizable
		CW_USEDEFAULT, CW_USEDEFAULT, m_width, m_height,
		nullptr, nullptr, hInstance, nullptr);

	if (hwnd == nullptr) {
		MessageBox(nullptr, L"Window Creation Failed", L"Error",
			MB_ICONEXCLAMATION | MB_OK);
	}

	SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

	HDC hdcScreen = GetDC(hwnd); // Obtain the screen/device context
	hdcBuffer = CreateCompatibleDC(hdcScreen); // Create a compatible device context for off-screen drawing

	BITMAPINFO bmi;
	ZeroMemory(&bmi, sizeof(bmi)); // Ensure the structure is initially empty
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = m_width; // Specify the width of the bitmap
	bmi.bmiHeader.biHeight = -m_height; // Negative height for a top-down DIB
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 24; // 24 bits per pixel (RGB)
	bmi.bmiHeader.biCompression = BI_RGB; // No compression

	hbmBuffer = CreateDIBSection(hdcBuffer, &bmi, DIB_RGB_COLORS, &pvBits, NULL, 0);
	SelectObject(hdcBuffer, hbmBuffer);
	ReleaseDC(hwnd, hdcScreen);

	ShowWindow(hwnd, SW_SHOWDEFAULT); // or use WS_VISIBLE but more control with this option
}

void Window::Render()
{
	if (m_renderCallback != nullptr &&
		m_renderSetting != nullptr &&
		m_densityGrid != nullptr)
	{
		unsigned char* pixel = static_cast<unsigned char*>(pvBits);
		fps = m_renderCallback(*m_densityGrid, *m_renderSetting, pixel);

		InvalidateRect(hwnd, NULL, TRUE);
		UpdateWindow(hwnd);
	}
}

void Window::OnMouseReleased()
{
	freecam_model.move_type = FreeCameraModel::CameraMoveType::NONE;
}

void Window::OnMousePressed(int flags, Vec2i location)
{
	freecam_model.mouse_pos = location;
	if (flags & EF_ALT_DOWN) {
		freecam_model.move_type =
			(flags & EF_LEFT_BUTTON_DOWN) ? FreeCameraModel::CameraMoveType::TUMBLE :
			(flags & EF_MIDDLE_BUTTON_DOWN) ? FreeCameraModel::CameraMoveType::TRACK :
			(flags & EF_RIGHT_BUTTON_DOWN) ? FreeCameraModel::CameraMoveType::DOLLY :
			(assert(false), FreeCameraModel::CameraMoveType::NONE);
	}
}

void Window::OnMouseMoved(const Vec2i& location)
{
	Vec2i delta = location - freecam_model.mouse_pos;
	freecam_model.mouse_pos = location;
	if (freecam_model.move_type == FreeCameraModel::CameraMoveType::NONE)
		return;
	switch (freecam_model.move_type) {
	case FreeCameraModel::CameraMoveType::TUMBLE: {
		freecam_model.theta -= delta.x() * kRotateAmplitude;
		freecam_model.phi -= delta.y() * kRotateAmplitude;
		UpdateCameraRotation();
	} break;
	case FreeCameraModel::CameraMoveType::DOLLY:
		OnMouseWheel(delta.x() + delta.y());
		return;
	case FreeCameraModel::CameraMoveType::TRACK: {
		Vec3<float> target_offset = multDirMatrix(
			Vec3<float>(-kPanAmplitude * delta.x(), kPanAmplitude * delta.y(), 0),
			rotation_mat
		);
		freecam_model.look_at += target_offset;
	}
											   break;
	default:
		break;
	}
	SetCameraMatrices();
}

void Window::OnMouseWheel(int scroll_amount)
{
	freecam_model.distance_to_target -= scroll_amount * kScrollAmplitude;
	SetCameraMatrices();
}

void Window::SetCameraMatrices()
{
	Vec3<float> camera_orient = multDirMatrix(Vec3f(0, 0, 1), rotation_mat);
	Vec3<float> camera_position = freecam_model.look_at +
		freecam_model.distance_to_target * camera_orient;
	cam_to_world = rotation_mat;
	cam_to_world[0][3] = camera_position[0];
	cam_to_world[1][3] = camera_position[1];
	cam_to_world[2][3] = camera_position[2];

	m_renderSetting->camera.setTransform(cam_to_world);

	Render();
}

void Window::UpdateCameraRotation()
{
	Quat<float> q1; q1.setAxisAngle(Vec3<float>(1, 0, 0), freecam_model.phi);
	Quat<float> q2; q2.setAxisAngle(Vec3<float>(0, 1, 0), freecam_model.theta);
	freecam_model.camera_rotation = q2 * q1;
	rotation_mat = quatToMatrix44(freecam_model.camera_rotation);
}
